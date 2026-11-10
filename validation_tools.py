"""Small, functional building blocks for a future collector-specific model.

These functions are exercised by numerical_checks(). They are not a fitted
model and do not assert that rear-surface sensors represent a single node.
"""
import numpy as np

def mass_flow(volume_lph, density=1040.0):
    """Convert measured volumetric L/h to mass kg/s."""
    return np.asarray(volume_lph) * density / 3.6e6

def delivered_heat(volume_lph, outlet_C, inlet_C, cp=3650.0, density=1040.0):
    """Signed enthalpy flux in W; retain negative heat exchange."""
    return mass_flow(volume_lph,density) * cp * (
        np.asarray(outlet_C)-np.asarray(inlet_C))

def thermal_rhs(t, temperatures, parameters, boundary):
    """Three-state wet-collector hypothesis with time-varying measured inputs.

    boundary(t) returns G [W/m2], Ta [C], Tin [C], flow [L/h].
    All physical parameters must be independently obtained or calibrated;
    there are deliberately no arbitrary defaults copied from the thesis.
    Required keys: A, alpha, eta_ref, beta, T_ref, H_ca, H_cb, H_ba, H_bw,
    C_c, C_b, C_f, cp, rho. H values are W/K; C values are J/K.
    This hypothesis omits explicit wind and radiation; assess residuals before
    interpreting its parameters. Efficiency must remain physically admissible
    throughout the calibrated temperature range; reject invalid parameters.
    """
    p=parameters
    Tc,Tb,Tf=temperatures
    G,Ta,Tin,volume_lph=boundary(t)
    eta=p['eta_ref']*(1-p['beta']*(Tc-p['T_ref']))
    if not 0 <= eta <= p['alpha'] <= 1:
        raise ValueError('Invalid efficiency/absorptance in fitted domain.')
    Pe=eta*p['A']*G
    qcb=p['H_cb']*(Tc-Tb)
    # Turning the pump off stops exported enthalpy, not solid-fluid conduction.
    qbf=p['H_bw']*(Tb-Tf)
    qout=float(delivered_heat(volume_lph,Tf,Tin,p['cp'],p['rho']))
    return np.array([
        (p['alpha']*p['A']*G-Pe-p['H_ca']*(Tc-Ta)-qcb)/p['C_c'],
        (qcb-p['H_ba']*(Tb-Ta)-qbf)/p['C_b'],
        (qbf-qout)/p['C_f']])

def error_scores(measured,predicted):
    """Pointwise errors in the units of the output; positive bias is overprediction.

    Feed independent held-out predictions, not calibration residuals. Do not
    combine temperature and electrical power into one dimensioned RMSE.
    """
    y=np.asarray(measured,dtype=float)
    pred=np.asarray(predicted,dtype=float)
    if y.shape!=pred.shape or y.size==0:raise ValueError('Invalid shapes.')
    if not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Flag missing samples explicitly before scoring.')
    e=pred-y
    return {'RMSE':float(np.sqrt(np.mean(e*e))),
        'MAE':float(np.mean(np.abs(e))), 'bias':float(np.mean(e))}

def numerical_checks():
    # Known dimensional checks, independent of any fitted collector parameters.
    assert np.isclose(mass_flow(60),0.017333333333333333)
    assert np.isclose(delivered_heat(60,45,40),316.3333333333333)
    assert np.isclose(delivered_heat(60,40,45),-316.3333333333333)
    assert delivered_heat(0,45,40)==0
    assert error_scores([1,2],[1,2])=={'RMSE':0.,'MAE':0.,'bias':0.}
    # Arbitrary states below test algebra only; they are not physical estimates.
    p=dict(A=1,alpha=.9,eta_ref=.2,beta=.004,T_ref=25,
           H_ca=10,H_cb=50,H_ba=5,H_bw=100,C_c=1000,C_b=2000,
           C_f=3000,cp=3650,rho=1040)
    b=lambda t:(500,20,30,0)
    y=np.array([45,40,35])
    rate=thermal_rhs(0,y,p,b)
    assert rate[2]>0  # Stagnant fluid remains thermally connected.
    stored=np.dot([p['C_c'],p['C_b'],p['C_f']],rate)
    eta=p['eta_ref']*(1-p['beta']*(y[0]-p['T_ref']))
    external=p['alpha']*500-eta*500-10*(45-20)-5*(40-20)
    assert np.isclose(stored,external,atol=1e-10)
    print('Six unit/sign/error checks and wet zero-flow energy checks passed.')

if __name__=='__main__':numerical_checks()
