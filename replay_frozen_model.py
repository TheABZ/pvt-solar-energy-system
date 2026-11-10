"""Portable SciPy replay of the first held-out date; no C compiler required.

Run: python replay_frozen_model.py
Place the checksum-verified data.csv here first, using the included downloader.
The initial state is the frozen model's pre-test forecast, not a measured test
target. This replay checks saved parameters; it does not repeat calibration.
"""
import json
import numpy as np
from scipy.integrate import solve_ivp
import experimental_benchmark as model

def main():
    _,dates,u,obs=model.load()
    frozen=json.loads((model.OUT/'frozen_model.json').read_text())
    p=np.array([frozen['chosen']['parameters'][n] for n in model.NAMES])
    e=np.array(frozen['electrical_coefficients'])
    start=46*17280;stop=start+17280
    inputs=u[start:stop];times=np.arange(len(inputs))*5.
    sol=solve_ivp(lambda t,y:model.rhs(t,y,p,e,inputs),(0,times[-1]),
        frozen['first_test_state_C'],method='LSODA',t_eval=times,
        max_step=2.5,rtol=1e-8,atol=1e-9)
    if not sol.success:raise RuntimeError(sol.message)
    mask=inputs[:,0]>=200
    flowing=mask&(inputs[:,3]>1040*3650/3.6e6)
    result={'date_utc':str(dates[start].date()),
       'PV_temperature_C':model.metrics(obs[start:stop,0][mask],sol.y[0,mask]),
       'flowing_outlet_C':model.metrics(obs[start:stop,1][flowing],sol.y[2,flowing])}
    print(json.dumps(result,indent=2))
    (model.OUT/'portable_replay_results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
