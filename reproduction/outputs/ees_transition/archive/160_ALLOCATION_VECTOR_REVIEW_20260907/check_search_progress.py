"""Independently reconcile completed diagnostic files; never modify the run."""
from pathlib import Path
import gzip, hashlib, json, time

A=Path(__file__).resolve().parent
R=A.parent/'157_PROPORTIONAL_SEARCH_AUDIT_20260907/v2'
p=json.loads((R/'AUDIT_PROTOCOL.json').read_text(encoding='utf-8'))
sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
binding=sha(R/'AUDIT_PROTOCOL.json')
complete=[]
for path in sorted((R/'days').glob('*/RESULT.json')):
    r=json.loads(path.read_text(encoding='utf-8'))
    assert r['protocol_sha256']==binding
    assert r['date'] in p['dates'] and r['date']==path.parent.name
    for n,digest in r['outputs_sha256'].items():
        assert sha(path.parent/n)==digest
    reproduction=json.loads((path.parent/'REPRODUCTION_CLASSIFICATION.json').read_text(encoding='utf-8'))
    assert reproduction['scan_ready'] and reproduction['allocation_and_screen_outputs_match']
    assert not reproduction['decision_errors']
    assert reproduction['captured_rays_sha256']==sha(path.parent/'CAPTURED_RAYS.npz')
    assert reproduction['strict_report_sha256']==sha(path.parent/'REPRODUCTION_STRICT.json')
    assert reproduction['all_fields_passed_original_strict_test']==r['all_fields_passed_original_strict_test']
    assert reproduction['auxiliary_loss_max_difference_kw']==r['auxiliary_loss_max_difference_kw']
    with gzip.open(path.parent/'RAY_SCANS.json.gz','rt',encoding='utf-8') as f:
        rays=json.load(f)
    keys={(x['interval_index'],x['control']) for x in rays}
    assert len(keys)==len(rays)==240
    assert keys=={(t,c) for t in range(48) for c in ('NO_FEEDBACK','PERSISTENCE_ONLY','ECF','REPLAY','MATCHED_UNIFORM')}
    counts=dict(scanned_rays=0,grid_ac_solves=0,rays_with_missed_larger_sample=0,
                rays_with_feasible_reentry=0,original_factor_failed_recheck=0)
    for x in rays:
        if x['scope']=='full_request_endpoint_already_maximal':
            assert x['original_factor']==1
            continue
        counts['scanned_rays']+=1
        points=x['points']
        rho=[q['rho'] for q in points]
        assert rho==sorted(set(rho)) and all(0<=v<=1 for v in rho)
        assert set(i/64 for i in range(65)).issubset(set(rho))
        assert x['original_factor'] in rho
        for q in points+x['reverse_recheck']:
            independent=bool(q['converged'] and q['undervoltage_count']==0 and
                             q['voltage_max_pu']<=1.07+1e-9 and q['transformer_overload_count']==0)
            assert q['feasible']==independent
        best=max((q['rho'] for q in points if q['feasible']),default=0.)
        assert best==x['largest_sampled_feasible']
        original=[q for q in points if q['rho']==x['original_factor']]
        assert len(original)==1 and original[0]['feasible']==x['original_factor_recheck_pass']
        assert (best>x['original_factor']+2e-5)==x['detected_miss']
        reentry=any(not a['feasible'] and b['feasible'] for a,b in zip(points,points[1:]))
        counts['rays_with_missed_larger_sample']+=int(x['detected_miss'])
        counts['rays_with_feasible_reentry']+=int(reentry)
        counts['original_factor_failed_recheck']+=int(not x['original_factor_recheck_pass'])
        counts['grid_ac_solves']+=len(points)+len(x['reverse_recheck'])
    for k,v in counts.items():assert r[k]==v,(r['date'],k)
    complete.append(r)
result=dict(snapshot_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            protocol_sha256=binding, completed_days=len(complete),expected_days=len(p['dates']),
            complete_dates=[r['date'] for r in complete],
            total_rays=sum(r['total_rays'] for r in complete),
            scanned_rays=sum(r['scanned_rays'] for r in complete),
            grid_ac_solves=sum(r['grid_ac_solves'] for r in complete),
            missed_larger_sample=sum(r['rays_with_missed_larger_sample'] for r in complete),
            feasible_reentry=sum(r['rays_with_feasible_reentry'] for r in complete),
            original_factor_failed_recheck=sum(r['original_factor_failed_recheck'] for r in complete),
            max_loss_reproduction_difference_kw=max((r['auxiliary_loss_max_difference_kw'] for r in complete),default=None),
            full_primary_coverage=len(complete)==80,
            interpretation='Completed files only. Independent hash, grid-coverage, predicate and count checks; no global-optimality or incomplete-day claim.')
(A/'SEARCH_PROGRESS_SNAPSHOT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False))
