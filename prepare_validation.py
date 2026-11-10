"""Download, verify and audit the Veynandt et al. experimental PV/T dataset.

Run: python prepare_validation.py --download
No thermal model is fitted and no experimental-validation claim is made.
Dependencies: numpy, pandas, matplotlib. Python >=3.10.
"""
from pathlib import Path
import argparse
import hashlib
import json
import urllib.request
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
URL = 'https://people.hochschule-burgenland.at/bitstreams/efa2346c-077c-4c88-b422-9a89f9fc34eb/download'
EXPECTED_MD5 = 'd0898b4ab924f4646a31571570fedd97'
CP, RHO = 3650.0, 1040.0  # Reported water/propylene-glycol properties, SI.

def digest(path):
    # Stream the large file rather than reading its bytes into RAM twice.
    h = hashlib.md5()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_data(path):
    # Reject altered input before parsing, so every QC result has provenance.
    if digest(path) != EXPECTED_MD5:
        raise ValueError('CSV checksum does not match the published repository.')
    frame = pd.read_csv(path)
    # Authors state UTC; parsing must not silently apply the local timezone.
    frame['date_time'] = pd.to_datetime(frame['date_time'],
        format='%d-%b-%Y %H:%M:%S', utc=True, errors='raise')
    if not frame['date_time'].is_monotonic_increasing:
        raise ValueError('Timestamps are not monotonically increasing.')
    if frame['date_time'].duplicated().any():
        raise ValueError('Repeated timestamps require explicit resolution.')
    return frame.set_index('date_time')

def annotate(frame):
    # Preserve every measured value. These columns are diagnostics, not edits.
    d = frame.copy()
    # L/h * (1e-3 m3/L) / (3600 s/h) * kg/m3 = kg/s.
    d['mass_flow_kgps'] = d['flow_rate_fluid_lph'] * RHO / 3.6e6
    # Signed thermal power: a cold collector can absorb heat from its loop.
    d['Q_fluid_W'] = d['mass_flow_kgps'] * CP * (
        d['T_fluid_out_C'] - d['T_fluid_in_C'])
    d['P_raw_W'] = d['I_PV_A'] * d['U_PV_V']
    d['P_filter_difference_W'] = d['P_el_W'] - d['P_raw_W']
    # Three positions do not define a measured area-weighted mean.
    rear = d[['T_back_surf_C','T_back_surf_left_C','T_back_surf_right_C']]
    d['rear_sensor_spread_K'] = rear.max(axis=1) - rear.min(axis=1)
    # The 200 W/m2 and 1 L/h thresholds define reporting strata only.
    d['daylight_stratum'] = d['I_g_t_Wpm2'] >= 200.0
    d['flowing_stratum'] = d['flow_rate_fluid_lph'] > 1.0
    d['illuminated_flowing_stratum'] = d['daylight_stratum'] & d['flowing_stratum']
    # Whole UTC dates prevent neighboring five-second rows leaking across sets.
    dates = pd.Index(d.index.floor('D').unique()).sort_values()
    n1, n2 = int(0.6 * len(dates)), int(0.8 * len(dates))
    labels = ['calibration']*n1 + ['development']*(n2-n1) + ['held_out']*(len(dates)-n2)
    partition = pd.DataFrame({'date_utc': dates, 'partition': labels})
    lookup = dict(zip(dates, labels))
    d['partition'] = d.index.floor('D').map(lookup)
    return d, partition

def analyse(d, partition, out):
    out.mkdir(exist_ok=True)
    numeric = d.select_dtypes(include=[np.number])
    steps = d.index.to_series().diff().dt.total_seconds().dropna()
    daylight = d['daylight_stratum']
    active = d['illuminated_flowing_stratum']
    def bounds(series):
        return {'min':float(series.min()), 'max':float(series.max())}
    stats = {
        'status':'Dataset assessment only; no fitted or validated model',
        'csv_bytes':(ROOT/'data.csv').stat().st_size,
        'md5':EXPECTED_MD5,
        'rows':len(d), 'original_numeric_columns':22,
        'start_utc':d.index[0].isoformat(), 'end_utc':d.index[-1].isoformat(),
        'unique_utc_dates':len(partition),
        'step_seconds_counts':{str(k):int(v) for k,v in steps.value_counts().items()},
        'missing_numeric_cells':int(numeric.isna().sum().sum()),
        'infinite_numeric_cells':int(np.isinf(numeric.to_numpy()).sum()),
        'daylight_rows':int(daylight.sum()),'illuminated_flowing_rows':int(active.sum()),
        'zero_flow_rows':int((d.flow_rate_fluid_lph==0).sum()),
        'negative_flow_rows':int((d.flow_rate_fluid_lph<0).sum()),
        'negative_tilted_irradiance_rows':int((d.I_g_t_Wpm2<0).sum()),
        'flow_on_transitions':int((d.flowing_stratum & ~d.flowing_stratum.shift(fill_value=False)).sum()),
        'pv_temperature_C':bounds(d.T_PV_C),
        'flow_Lph':bounds(d.flow_rate_fluid_lph),
        'tilted_irradiance_Wpm2':bounds(d.I_g_t_Wpm2),
        'inlet_temperature_C':bounds(d.T_fluid_in_C),
        'ambient_temperature_C':bounds(d.T_air_C),
        'rear_spread_daylight_median_K':float(d.loc[daylight,'rear_sensor_spread_K'].median()),
        'rear_spread_daylight_p95_K':float(d.loc[daylight,'rear_sensor_spread_K'].quantile(.95)),
        'power_difference_gt1W_daylight_rows':int((daylight & (d.P_filter_difference_W.abs()>1)).sum()),
        'negative_heat_active_rows':int((active & (d.Q_fluid_W<0)).sum()),
        'partitions':{},
    }
    for label, dates in partition.groupby('partition',sort=False):
        mask = d.partition==label
        stats['partitions'][label] = {'days':len(dates), 'rows':int(mask.sum()),
            'first_date':dates.date_utc.min().strftime('%Y-%m-%d'),
            'last_date':dates.date_utc.max().strftime('%Y-%m-%d')}
    # Whole-dataset descriptive QC is not a validation score. No parameter fit.
    daily = d.resample('D').agg(
        samples=('T_PV_C','size'), pv_max_C=('T_PV_C','max'),
        inlet_mean_C=('T_fluid_in_C','mean'), irradiance_peak_Wpm2=('I_g_t_Wpm2','max'),
        flow_mean_Lph=('flow_rate_fluid_lph','mean'),
        daylight_samples=('daylight_stratum','sum'),
        illuminated_flowing_samples=('illuminated_flowing_stratum','sum'),
        rear_spread_p95_K=('rear_sensor_spread_K',lambda x:x.quantile(.95)))
    daily['partition'] = d['partition'].resample('D').first()
    daily.to_csv(out/'daily_quality_summary.csv',index_label='date_utc')
    partition.to_csv(out/'chronological_partitions.csv', index=False)
    d.select_dtypes(include=[np.number]).describe(percentiles=[.01,.05,.5,.95,.99]).T.to_csv(out/'column_statistics.csv')
    (out/'quality_results.json').write_text(json.dumps(stats,indent=2))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(3,1,figsize=(10,8),sharex=True,layout='constrained')
    colors={'calibration':'#156082','development':'#c78c20','held_out':'#687d3f'}
    for lab, g in daily.groupby('partition',sort=False):
        axes[0].plot(g.index,g.pv_max_C,'.-',color=colors[lab],label=lab.replace('_',' '))
    axes[0].set_ylabel('Daily maximum PV\nsensor temperature (°C)');axes[0].legend(ncols=3,fontsize=9)
    axes[1].plot(daily.index,daily.irradiance_peak_Wpm2,color='#c78c20')
    axes[1].set_ylabel('Peak plane-of-array\nirradiance (W/m²)')
    axes[2].plot(daily.index,daily.illuminated_flowing_samples/720,color='#156082')
    axes[2].set_ylabel('Illuminated, flowing\ntime (hours/day)');axes[2].set_xlabel('Date (UTC)')
    fig.suptitle('Measured dataset coverage and proposed chronological partitions')
    fig.savefig(out/'dataset_coverage.png',dpi=220);plt.close(fig)

    # Illustrative detail is deliberately selected from calibration dates only.
    chosen = daily.loc[daily.partition=='calibration','illuminated_flowing_samples'].idxmax()
    day = d.loc[d.index.floor('D')==chosen]
    # Minute means are for rendering only; integration should retain 5 s inputs.
    v = day.select_dtypes(include=[np.number]).resample('1min').mean()
    hours = (v.index-v.index[0]).total_seconds()/3600
    fig,ax=plt.subplots(4,1,figsize=(10,9),sharex=True,layout='constrained')
    ax[0].plot(hours,v.I_g_t_Wpm2,color='#c78c20',label='Plane-of-array irradiance')
    ax[0].set_ylabel('W/m²');ax[0].legend(loc='upper left')
    for col,label in [('T_PV_C','PV rear sensor'),('T_back_surf_C','Central rear surface'),('T_fluid_in_C','Fluid inlet'),('T_fluid_out_C','Fluid outlet')]:
        ax[1].plot(hours,v[col],label=label)
    ax[1].set_ylabel('Temperature (°C)');ax[1].legend(ncols=2,fontsize=8)
    ax[2].plot(hours,v.flow_rate_fluid_lph,color='#156082');ax[2].set_ylabel('Flow (L/h)')
    ax[3].plot(hours,v.P_raw_W,label='Raw I × V',alpha=.65)
    ax[3].plot(hours,v.P_el_W,label='Published filtered power',alpha=.85)
    ax[3].set_ylabel('Electrical power (W)');ax[3].set_xlabel('Hour (UTC)');ax[3].legend(fontsize=8)
    fig.suptitle(f'Measured calibration-day example: {chosen:%d %B %Y}')
    fig.savefig(out/'measured_day.png',dpi=220);plt.close(fig)
    stats['illustrative_calibration_day']=chosen.strftime('%Y-%m-%d')
    (out/'quality_results.json').write_text(json.dumps(stats,indent=2))
    return stats

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    path=ROOT/'data.csv'
    if not path.exists():
        if not args.download: raise FileNotFoundError('Use --download to obtain the public CSV.')
        temp=path.with_suffix('.partial')
        with urllib.request.urlopen(URL,timeout=120) as response, temp.open('wb') as dest:
            for block in iter(lambda: response.read(1024*1024),b''):dest.write(block)
        if digest(temp)!=EXPECTED_MD5:raise ValueError('Downloaded file failed checksum.')
        temp.replace(path)
    d, partition=annotate(load_data(path))
    print(json.dumps(analyse(d,partition,ROOT/'outputs'),indent=2))

if __name__=='__main__':main()
