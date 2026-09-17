"""Pre-run input checks; no new-factor power flow or outcome selection."""
from pathlib import Path
import json
import numpy as np
import run_scale as run
from scale_inputs import InputReference,near
A=Path(__file__).resolve().parent

def main():
    p=run.verify_sources();ref=InputReference(p);date=p['dates'][0]
    primary=ref.expected(1.,date);cases=[];rejected=[]
    for setting in p['settings']:
        f=setting['factor'];e=ref.expected(f,date)
        assert np.allclose(e['capacity'],primary['capacity']*f,rtol=0,atol=1e-12)
        assert np.array_equal(e['load'],primary['load'])
        assert np.allclose(e['pv'],primary['pv']*f,rtol=0,atol=1e-11)
        assert np.all(e['raw']<=e['capacity']+1e-12) and np.all(e['raw']>=0)
        assert np.all(e['raw'][:,~ref.part]==0) and np.all(e['available'][:,~ref.part]==0)
        m=run.load_runner(f)
        assert m.PV_SCALE==p['original_pv_scale']*f
        assert m.REPLAY_TOLERANCE_MW==2e-6 and m.AUTHORIZATION_VOLTAGE_MAX_PU==1.07
        m._worker_init(dict(p['config']))
        worker=m._WORKER
        assert np.array_equal(worker['participant'],ref.part)
        near(worker['capacity_connection_mw'],e['capacity'])
        near(worker['upper_residual'],e['increment'][:,worker['used_profile_index']])
        for label,actual,wrong in [
            ('old_capacity',e['capacity'],primary['capacity']),
            ('old_forecast_increment',e['increment'],primary['increment']),
            ('old_request',e['raw'],primary['raw']),
            ('scaled_demand',e['load'],primary['load']*f),
            ('scaled_net_availability',e['available'],primary['available']*f)]:
            try:near(actual,wrong)
            except AssertionError:rejected.append(setting['id']+':'+label)
            else:raise AssertionError('Negative test failed to reject '+label)
        cases.append(dict(setting=setting['id'],factor=f,participant_capacity_MW=float(e['capacity'].sum()),first_day_raw_request_MWh=float(e['raw'].sum()*.5),first_day_available_export_MWh=float(e['available'].sum()*.5)))
    # A capacity change acts before subtracting demand; it is not a record multiplier.
    assert max(2*2-1,0)!=2*max(2-1,0)
    report=dict(passed=True,settings=cases,source_coefficients_use_earlier_year_only=True,
                no_new_factor_power_flows=True,load_unchanged=True,nonparticipant_mask_verified=True,
                negative_or_physical_outcomes_not_used=True,rejected_inconsistent_inputs=rejected,
                original_runner_inputs_cross_checked=True,protocol_sha256=run.sha(A/'PROTOCOL.json'))
    (A/'INPUT_TESTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
