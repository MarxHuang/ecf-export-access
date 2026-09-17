from pathlib import Path
import json,time
import run_scale as run
A=Path(__file__).resolve().parent
B=A.parent/'180_PROFILE_MAPPING_ROBUSTNESS_20260908'
FILES=['prepare_scale.py','run_scale.py','scale_inputs.py','test_scale.py','audit_replay_ac.py','verify_scale.py','pipeline.py','PROTOCOL.md']

def main():
    assert not (A/'PROTOCOL.json').exists(),'Frozen protocol already exists'
    old=json.loads((B/'PROTOCOL.json').read_text('utf-8'))
    source=dict(old['source_hashes'])
    source[str(B/'PROTOCOL.json')]=run.sha(B/'PROTOCOL.json')
    for f,h in source.items():assert run.sha(Path(f))==h,f
    settings=[dict(id=n,factor=f) for n,f in [('PV050',.5),('PV075',.75),('PV125',1.25),('PV150',1.5)]]
    cfg=dict(old['config']);assert cfg['load_scale']==.3 and cfg['beta']==.5 and cfg['telemetry_mode']=='EXACT'
    p=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),dates=old['dates'],config=cfg,settings=settings,
           source_hashes=source,implementation_hashes={n:run.sha(A/n) for n in FILES},
           original_pv_scale=run.load_runner().PV_SCALE,expected_new_days=320,expected_control_intervals=61440,
           expected_replays=15360,participant_connections=418,connection_count=1255,
           primary_already_observed=True,controller_and_AC_source_unchanged=True)
    assert len(p['dates'])==80
    (A/'PROTOCOL.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(frozen=True,settings=settings,protocol_sha256=run.sha(A/'PROTOCOL.json'),input_files=len(source))))

if __name__=='__main__':main()
