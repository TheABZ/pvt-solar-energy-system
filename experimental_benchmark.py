"""Reproducible conditional experimental benchmark, not a paired cooling test.

Run after obtaining the public CSV with prepare_validation.py --download.
Linux/macOS: install numpy scipy pandas matplotlib; a C compiler is required.
The small C source accelerates repeated integration; all research operations
are ordinary Python functions. No physical parameter is claimed uniquely known.
"""
from pathlib import Path
import ctypes, subprocess, json, hashlib, time, platform
import numpy as np
import pandas as pd
from scipy.optimize import least_squares,lsq_linear
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'experimental_outputs';OUT.mkdir(exist_ok=True)
CSV=ROOT/'data.csv'
if not CSV.exists():CSV=ROOT.parent/'pvt_validation/data.csv'
NAMES=['C_PV','C_absorber','C_fluid','H_front','H_PV_absorber',
       'H_absorber_rear','H_absorber_fluid','H_sky','solar_aperture','H_wind']
LOW=np.array([200,200,200,.1,1,.1,1,.1,.4,.001])
HIGH=np.array([200000,500000,100000,200,2000,200,3000,30,3.,50.])
START=np.array([12000,12000,3650,12,120,5,160,6,1.6,2])
CP,RHO=3650.,1040.

def load():
    h=hashlib.md5()
    with CSV.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!='d0898b4ab924f4646a31571570fedd97':raise ValueError('Source checksum mismatch')
    d=pd.read_csv(CSV)
    dates=pd.to_datetime(d.date_time,format='%d-%b-%Y %H:%M:%S',utc=True)
    if not (dates.diff().dropna().dt.total_seconds()==5).all():raise ValueError('Unexpected time spacing')
    # Use measured exterior rear temperature as a boundary, not internal state.
    u=np.ascontiguousarray(np.column_stack([d.I_g_t_Wpm2,d.T_air_C,d.T_fluid_in_C,
       d.flow_rate_fluid_lph*RHO/3.6e6*CP,d.T_back_surf_C,d.T_sky_C,d.v_wind_t_mps]))
    observed=np.column_stack([d.T_PV_C,d.T_fluid_out_C,d.P_el_W])
    return d,pd.DatetimeIndex(dates),u,observed

def compile_solver():
    lib=ROOT/'thermal_core.so'
    subprocess.run(['cc','-O3','-shared','-fPIC',str(ROOT/'thermal_core.c'),'-o',str(lib)],check=True)
    fn=ctypes.CDLL(str(lib)).simulate
    arr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,arr,arr,arr,arr,ctypes.c_double,ctypes.c_int,arr]
    fn.restype=None
    return fn

def integrate(fn,u,p,e,y0,sub=1):
    y=np.empty((len(u),3),dtype=np.float64)
    fn(len(u),np.ascontiguousarray(u),np.ascontiguousarray(p),
       np.ascontiguousarray(e),np.ascontiguousarray(y0),5.,sub,y)
    return y

def power(G,T,e):
    return G*(e[0]+e[1]*(T-25)+e[2]*(G-1000)/1000)

def metrics(y,pred):
    er=np.asarray(pred)-np.asarray(y)
    return dict(RMSE=float(np.sqrt(np.mean(er**2))),MAE=float(np.mean(abs(er))),bias=float(np.mean(er)))

def rhs(t,y,p,e,u):
    # Independent SciPy replay uses the same explicitly linear input interpolation.
    k=min(int(t//5),len(u)-2);w=(t-k*5)/5
    G,Ta,Tin,F,Tr,Ts,v=(1-w)*u[k]+w*u[k+1]
    Tc,Tb,Tf=y;H=p[3]+p[9]*v
    Pe=power(G,Tc,e)
    return [(p[8]*G-Pe-H*(Tc-Ta)-p[4]*(Tc-Tb)-p[7]*(Tc-Ts))/p[0],
       (p[4]*(Tc-Tb)-p[5]*(Tb-Tr)-p[6]*(Tb-Tf))/p[1],
       (p[6]*(Tb-Tf)-F*(Tf-Tin))/p[2]]

def main():
    start=time.monotonic();d,dates,u,obs=load();fn=compile_solver()
    ntrain=34*17280;nval=46*17280
    day=u[:,0]>=200;flow=d.flow_rate_fluid_lph.to_numpy()>1
    active=day&flow;train=np.arange(len(u))<ntrain
    # Electrical regression uses ONLY measured calibration data at G>=200.
    G=u[:,0];Tc=obs[:,0]
    X=np.column_stack([G,G*(Tc-25),G*(G-1000)/1000])
    mask=train&day
    ele=lsq_linear(X[mask],obs[mask,2],bounds=([.01,-.01,-.3],[.8,.01,.3]),tol=1e-12).x
    print('Electrical aperture coefficients:',ele,flush=True)
    y0=np.array([obs[0,0],(obs[0,0]+obs[0,1])/2,obs[0,1]])
    # Residual sampling is one minute; ALL 5 s input knots are integrated.
    ix=np.arange(1440,ntrain,12)  # Omit first 2 hours for latent initial-state relaxation.
    ipv=ix[day[ix]];ifl=ix[active[ix]]
    fits=[]
    for variant in ['constant_front','wind_front']:
        npar=9 if variant=='constant_front' else 10
        best=None
        for seed in [0,1]:
            x0=START.copy()
            if seed:x0[:8]*=np.array([1.8,.6,1.6,.6,1.5,1.8,.7,1.2])
            def unpack(x):
                p=np.exp(x)
                if npar==9:p=np.r_[p,0.]
                return p
            def residual(x):
                z=integrate(fn,u[:ntrain],unpack(x),ele,y0)
                return np.r_[z[ipv,0]-obs[ipv,0],z[ifl,2]-obs[ifl,1]]
            opt=least_squares(residual,np.log(x0[:npar]),bounds=(np.log(LOW[:npar]),np.log(HIGH[:npar])),
                max_nfev=90,ftol=2e-5,xtol=2e-5,gtol=2e-4,diff_step=1e-4)
            p=unpack(opt.x);cost=float(np.mean(residual(opt.x)**2))
            record=dict(variant=variant,seed=seed,training_MSE=cost,nfev=opt.nfev,
                success=bool(opt.success),message=opt.message,parameters=dict(zip(NAMES,map(float,p))),
                active_bounds=[NAMES[i] for i,v in enumerate(opt.active_mask) if v!=0],
                jacobian_condition=float(np.linalg.cond(opt.jac)))
            fits.append(record)
            print('Fit:',variant,seed,'MSE',cost,'nfev',opt.nfev,flush=True)
            if best is None or cost<best[0]:best=(cost,p,record)
        # Model development is restricted to days 35-46. Do not inspect test scores yet.
        z=integrate(fn,u[:nval],best[1],ele,y0)
        dev=(np.arange(nval)>=ntrain)
        pv=metrics(obs[:nval,0][dev&day[:nval]],z[:,0][dev&day[:nval]])
        fluid=metrics(obs[:nval,1][dev&active[:nval]],z[:,2][dev&active[:nval]])
        best[2]['development_PV']=pv;best[2]['development_fluid']=fluid
        best[2]['selection_score_K']=(pv['RMSE']+fluid['RMSE'])/2
    candidates=[f for f in fits if 'selection_score_K' in f]
    chosen=min(candidates,key=lambda r:r['selection_score_K'])
    (OUT/'frozen_model.json').write_text(json.dumps({'chosen':chosen,'electrical_coefficients':ele.tolist(),
      'selection':'Mean of development PV and flowing-outlet temperature RMSE; no test scores used'},indent=2))
    p=np.array([chosen['parameters'][n] for n in NAMES]);z=integrate(fn,u,p,ele,y0)
    frozen=json.loads((OUT/'frozen_model.json').read_text())
    frozen.update(initial_state_C=y0.tolist(),first_test_state_C=z[nval].tolist(),
      first_test_time_UTC=dates[nval].isoformat(),
      parameter_bounds={n:[float(lo),float(hi)] for n,lo,hi in zip(NAMES,LOW,HIGH)})
    (OUT/'frozen_model.json').write_text(json.dumps(frozen,indent=2))
    predpower=power(G,z[:,0],ele)
    # Simple static regressions are fitted on calibration measurements only.
    B=np.column_stack([np.ones(len(u)),G,u[:,1],u[:,2],u[:,4],u[:,6],u[:,3]])
    baselines=[]
    for target,m in [(0,train&day),(1,train&active)]:
        c=np.linalg.lstsq(B[m],obs[m,target],rcond=None)[0];baselines.append(B@c)
    reports={}
    for label,left,right in [('calibration',0,ntrain),('development',ntrain,nval),('held_out',nval,len(u))]:
        period=(np.arange(len(u))>=left)&(np.arange(len(u))<right)
        m=period&day;f=period&active
        reports[label]={'PV_temperature_C':metrics(obs[m,0],z[m,0]),
          'flowing_outlet_C':metrics(obs[f,1],z[f,2]),
          'electrical_power_W':metrics(obs[m,2],predpower[m]),
          'static_PV_temperature_C':metrics(obs[m,0],baselines[0][m]),
          'static_flowing_outlet_C':metrics(obs[f,1],baselines[1][f]),
          'PV_samples':int(m.sum()),'outlet_samples':int(f.sum())}
    # Grid refinement on the final model; compare continuous entire trajectories.
    fine=integrate(fn,u,p,ele,y0,sub=2)
    finer=integrate(fn,u,p,ele,y0,sub=4)
    refinement={'5_vs_2p5_s_max_C':np.max(abs(z-fine),axis=0).tolist(),
       '2p5_vs_1p25_s_max_C':np.max(abs(fine-finer),axis=0).tolist()}
    # Independent LSODA replay of 6 h within calibration, with all input knots resolved.
    a=14*17280+6*720;b=a+6*720+1;ud=u[a:b]
    tt=np.arange(len(ud))*5.
    sol=solve_ivp(lambda t,y:rhs(t,y,p,ele,ud),(0,tt[-1]),finer[a],t_eval=tt,
       method='LSODA',rtol=1e-8,atol=1e-9,max_step=2.5)
    if not sol.success:raise RuntimeError(sol.message)
    independent=integrate(fn,ud,p,ele,finer[a],sub=4)
    refinement['LSODA_vs_1p25_s_max_C']=np.max(abs(sol.y.T-independent),axis=0).tolist()
    # Signed heat metrics: only illuminated flowing intervals, not stagnant-pipe readings.
    qobs=u[:,3]*(obs[:,1]-u[:,2]);qpred=u[:,3]*(z[:,2]-u[:,2])
    for label,left,right in [('calibration',0,ntrain),('development',ntrain,nval),('held_out',nval,len(u))]:
        m=(np.arange(len(u))>=left)&(np.arange(len(u))<right)&active
        reports[label]['thermal_power_W']=metrics(qobs[m],qpred[m])
        # Explicit 5 s rectangular sum; stratum energy, not whole-day heat yield.
        reports[label]['signed_active_heat_measured_kWh']=float(np.sum(qobs[m])*5/3.6e6)
        reports[label]['signed_active_heat_predicted_kWh']=float(np.sum(qpred[m])*5/3.6e6)
    held=np.arange(len(u))>=nval
    daily=pd.DataFrame({'date_utc':dates,'G':G,'PV_observed_C':obs[:,0],
      'PV_predicted_C':z[:,0],'outlet_observed_C':obs[:,1],'outlet_predicted_C':z[:,2],
      'power_observed_W':obs[:,2],'power_predicted_W':predpower,
      'flow_Lph':d.flow_rate_fluid_lph}).set_index('date_utc')
    daily.resample('1min').mean().to_csv(OUT/'minute_predictions_for_plotting.csv.gz',compression='gzip')
    # Evaluation uses full 5 s arrays; minute data are a small plotting export only.
    rows=[]
    for name,g in daily.groupby(daily.index.floor('D')):
        rows.append({'date_utc':str(name.date()),'partition':'calibration' if name<dates[ntrain] else 'development' if name<dates[nval] else 'held_out',
          'PV_RMSE_C':metrics(g.PV_observed_C[g.G>=200],g.PV_predicted_C[g.G>=200])['RMSE']})
    pd.DataFrame(rows).to_csv(OUT/'daily_PV_errors.csv',index=False)
    results={'scope':'Conditional prediction benchmark; rear surface is a measured boundary, absorber is latent',
      'data_DOI':'10.57739/3720','working_fluid':'43% propylene glycol by mass in water',
      'Cp_JpkgK':CP,'density_kgpm3':RHO,'area':'Not fixed or inferred separately from absorptance',
      'electrical_aperture_coefficients':ele.tolist(),'empirical_relative_temperature_coefficient_per_K':float(ele[1]/ele[0]),
      'selected_variant':chosen['variant'],'selected_parameters':chosen['parameters'],
      'all_fit_attempts':fits,'scores':reports,'numerical_refinement':refinement,
      'parameter_physical_uniqueness':'Not established; no confidence intervals claimed',
      'elapsed_seconds':time.monotonic()-start,'python':platform.python_version()}
    # Independently assembled whole-network energy rate cancels internal fluxes.
    closure=[]
    for i in np.random.default_rng(21).integers(0,len(u)-2,200):
        state=z[i];Gv,Ta,Tin,F,Tr,Ts,v=u[i]
        stored=np.array(rhs(0,state,p,ele,u[i:i+2]))@p[:3]
        external=p[8]*Gv-power(Gv,state[0],ele)-(p[3]+p[9]*v)*(state[0]-Ta)\
          -p[5]*(state[1]-Tr)-p[7]*(state[0]-Ts)-F*(state[2]-Tin)
        closure.append(abs(stored-external))
    results['instantaneous_energy_cancellation_max_W']=float(max(closure))
    (OUT/'benchmark_results.json').write_text(json.dumps(results,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Choose first full test date, independent of achieved model accuracy.
    g=daily.loc[str(dates[nval].date())].resample('1min').mean()
    hh=(g.index-g.index[0]).total_seconds()/3600
    fig,ax=plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    ax[0].plot(hh,g.G,color='#b88013');ax[0].set_ylabel('Irradiance (W/m²)')
    ax[1].plot(hh,g.PV_observed_C,label='Measured PV rear sensor',color='#156082')
    ax[1].plot(hh,g.PV_predicted_C,'--',label='Predicted PV node',color='#b24736');ax[1].set_ylabel('PV temperature (°C)');ax[1].legend(fontsize=8)
    # Outlet predictions are interpreted only during the declared flowing stratum.
    mf=(g.flow_Lph>1)&(g.G>=200)
    ax[2].plot(hh,np.where(mf,g.outlet_observed_C,np.nan),label='Measured outlet')
    ax[2].plot(hh,np.where(mf,g.outlet_predicted_C,np.nan),'--',label='Predicted fluid node')
    ax[2].set_ylabel('Flowing outlet (°C)');ax[2].set_xlabel('Hour (UTC)');ax[2].legend(fontsize=8)
    fig.suptitle(f'Conditional held-out prediction: {dates[nval]:%d %B %Y}')
    fig.savefig(OUT/'held_out_day.png',dpi=220);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    for ax,target,pred,m,title in [(axes[0],obs[:,0],z[:,0],held&day,'PV rear temperature'),(axes[1],obs[:,1],z[:,2],held&active,'Flowing outlet temperature')]:
        ax.hexbin(target[m],pred[m],gridsize=50,mincnt=1,cmap='Blues',bins='log')
        lo=min(target[m].min(),pred[m].min());hi=max(target[m].max(),pred[m].max())
        ax.plot([lo,hi],[lo,hi],'k--',lw=1);ax.set_title(title);ax.set_xlabel('Measured (°C)');ax.set_ylabel('Predicted (°C)')
    fig.savefig(OUT/'held_out_parity.png',dpi=220);plt.close(fig)
    print(json.dumps({'selected':chosen['variant'],'scores':reports,'refinement':refinement,'seconds':time.monotonic()-start},indent=2),flush=True)

if __name__=='__main__':main()
