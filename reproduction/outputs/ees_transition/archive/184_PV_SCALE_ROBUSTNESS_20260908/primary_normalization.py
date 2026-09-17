"""Normalize existing primary outcomes without rerunning any control."""
from pathlib import Path
import json
import numpy as np
import run_scale as run
from scale_inputs import InputReference,near
A=Path(__file__).resolve().parent
F=run.F/'runs/primary'

def main():
    p=run.verify_sources();ref=InputReference(p)
    summary_path=F/'AU_EXTERNAL_RUN_SUMMARY.json'
    source_hashes={str(summary_path):run.sha(summary_path)}
    source=json.loads(summary_path.read_text('utf-8'))
    days={}
    for file in sorted((F/'day_checkpoints').glob('*.json')):
        r=json.loads(file.read_text('utf-8'))['result']
        date=r['day_summaries'][0]['date']
        if date in p['dates']:
            assert date not in days
            days[date]=r;source_hashes[str(file)]=run.sha(file)
    assert set(days)==set(p['dates'])
    totals={c:dict(delivered=0.,authorized=0.,unused=0.,available=0.,power_flow_count=0) for c in ['NO_FEEDBACK','ECF','MATCHED_UNIFORM','PERSISTENCE_ONLY']}
    a_total=0.;requested_available=0.;request_limited_intervals=0;positive_export_intervals=0
    for date in p['dates']:
        e=ref.expected(1.,date)
        a_total+=float(e['available'].sum()*.5)
        requested_available+=float(np.minimum(e['available'],e['raw']).sum()*.5)
        positive_export_intervals+=int((e['available'].sum(axis=1)>2e-6).sum())
        r=days[date]
        for c in totals:
            records=sorted([v for v in r['interval_records'] if v['control']==c],key=lambda v:v['interval_index'])
            assert len(records)==48
            out=totals[c]
            for target,key in [('delivered','delivered_export_kw'),('authorized','authorized_export_kw'),('unused','idle_authorization_kw'),('available','available_export_kw')]:
                out[target]+=sum(v[key] for v in records)*.5/1000
            out['power_flow_count']+=sum(v['power_flow_count'] for v in records)
            if c=='NO_FEEDBACK':
                near(np.array([v['raw_request_kw'] for v in records]),e['raw'].sum(axis=1)*1000,1e-7)
                request_limited_intervals+=sum(v['admitted_request_kw']-v['authorized_export_kw']>.002 for v in records)
    for c,out in totals.items():
        for name,key in [('delivered','delivered_export_mwh'),('authorized','authorized_export_mwh'),('unused','idle_authorization_mwh'),('available','available_export_mwh')]:
            near(out[name],source['aggregate'][c][key],1e-9)
        near(out['delivered'],out['authorized']-out['unused'],1e-9)
        near(out['available'],a_total,1e-9)
    cap=float(ref.expected(1.,p['dates'][0])['capacity'].sum())
    missing=a_total-totals['NO_FEEDBACK']['delivered']
    gain=totals['ECF']['delivered']-totals['NO_FEEDBACK']['delivered']
    request_gap=a_total-requested_available
    requested_gap=requested_available-totals['NO_FEEDBACK']['delivered']
    near(request_gap+requested_gap,missing,1e-9)
    assert request_gap>=0 and requested_gap>=0
    result=dict(primary_days=len(days),participant_capacity_MW=cap,available_MWh=a_total,
        no_feedback_undelivered_available_MWh=missing,
        available_above_no_feedback_requests_MWh=request_gap,
        requested_available_not_delivered_MWh=requested_gap,
        ECF_gain_MWh=gain,gain_MWh_per_installed_MW=gain/cap,
        gain_fraction_of_no_feedback_delivery=gain/totals['NO_FEEDBACK']['delivered'],
        gain_fraction_of_no_feedback_undelivered_available=gain/missing,
        positive_available_intervals=positive_export_intervals,
        no_feedback_request_limited_intervals=request_limited_intervals,totals=totals,
        interpretation='Undelivered available export is not all physically recoverable. It includes output above requests and output requested but not delivered. None of these denominators measures a causal opportunity set or field curtailment.',
        new_power_flows=0,source_hashes=source_hashes,protocol_sha256=run.sha(A/'PROTOCOL.json'))
    for file,h in source_hashes.items():assert run.sha(Path(file))==h
    run.verify_sources()
    (A/'PRIMARY_NORMALIZATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','totals']},indent=2))

if __name__=='__main__':main()
