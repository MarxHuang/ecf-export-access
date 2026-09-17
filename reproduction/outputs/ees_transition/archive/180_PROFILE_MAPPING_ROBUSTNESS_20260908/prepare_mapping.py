from pathlib import Path
import collections,csv,hashlib,importlib.util,json,time

A=Path(__file__).resolve().parent
F=A.parent/'151_FROZEN_SCENARIO_EVALUATION_20260907'
B=A.parent/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def mapping_check(original, altered):
    assert len(original)==len(altered)==1255
    assert len({r['load_name'] for r in altered})==1255
    for x,y in zip(original,altered):
        assert {k:v for k,v in x.items() if k!='ausgrid_customer_id'}=={k:v for k,v in y.items() if k!='ausgrid_customer_id'}
    for group in ['0','1']:
        a=collections.Counter(r['ausgrid_customer_id'] for r in original if r['participant_33pct']==group)
        b=collections.Counter(r['ausgrid_customer_id'] for r in altered if r['participant_33pct']==group)
        assert a==b
    assert sum(r['participant_33pct']=='1' for r in altered)==418

def main():
    assert not (A/'PROTOCOL.json').exists(),'Frozen protocol exists'
    old=json.loads((B/'PROTOCOL.json').read_text('utf-8'))
    cfg=dict(old['config']);cfg.update(telemetry_dropout_fraction=0.,telemetry_dropout_seed='')
    with Path(cfg['mapping_file']).open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    mapping_check(rows,rows)
    maps=[]
    for number,key in enumerate(range(18001,18006),1):
        altered=[dict(r) for r in rows]
        for group in ['0','1']:
            recipient=sorted([i for i,r in enumerate(rows) if r['participant_33pct']==group],key=lambda i:rows[i]['load_name'])
            donors=sorted(recipient,key=lambda i:(hashlib.sha256(f"{key}|{group}|{rows[i]['load_name']}".encode()).hexdigest(),rows[i]['load_name']))
            for i,j in zip(recipient,donors): altered[i]['ausgrid_customer_id']=rows[j]['ausgrid_customer_id']
        mapping_check(rows,altered)
        name=f'M{number:02}'
        file=A/(name+'.csv')
        with file.open('x',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(altered)
        maps.append(dict(id=name,key=key,path=str(file),sha256=sha(file),changed_profiles=sum(x['ausgrid_customer_id']!=y['ausgrid_customer_id'] for x,y in zip(rows,altered))))
    source=dict(old['inputs'])
    for n in ['PROTOCOL.json','PROTOCOL.md','run_comparison.py']: source[str(B/n)]=sha(B/n)
    for name in ['AU_EXTERNAL_DAY_SUMMARY.csv','AU_EXTERNAL_INTERVAL_RESULTS.csv']:
        file=F/'runs/primary'/name;source[str(file)]=sha(file)
    for file,h in source.items(): assert sha(Path(file))==h,file
    runner=F/'scientific_source/tools/run_ees_australian_external_replay.py'
    spec=importlib.util.spec_from_file_location('mapping_source_runner',runner)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    connections=m.parse_dss_load_connections(Path(cfg['dss_root']))
    original_indices=m._eligible_days(Path(cfg['package']),'EVALUATION',connections,Path(cfg['mapping_file']))
    for r in maps: assert m._eligible_days(Path(cfg['package']),'EVALUATION',connections,Path(r['path']))==original_indices
    p=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),dates=old['dates'],config=cfg,mappings=maps,
           source_hashes=source,implementation_hashes={n:sha(A/n) for n in ['prepare_mapping.py','run_mapping.py','test_mapping.py','PROTOCOL.md']},
           rows_per_mapping=1255,participant_connections=418,days_per_mapping=80,expected_new_days=400,
           expected_control_intervals=76800,expected_replays=19200,controls=list(m.CONTROLS),primary_already_observed=True)
    assert len(old['dates'])==80 and len(set(r['sha256'] for r in maps))==5
    (A/'PROTOCOL.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'frozen':True,'maps':maps,'source_files':len(source),'protocol_sha256':sha(A/'PROTOCOL.json')}))

if __name__=='__main__':main()
