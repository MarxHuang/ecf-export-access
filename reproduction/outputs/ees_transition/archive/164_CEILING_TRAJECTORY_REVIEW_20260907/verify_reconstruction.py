"""Independent NumPy equation and category checks of reconstructed arrays."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
A=Path(__file__).resolve().parent;F=A.parent/'151_FROZEN_SCENARIO_EVALUATION_20260907/runs'
manifest=json.loads((A/'RECONSTRUCTION.json').read_text('utf-8'))
case={'0.25':'beta_025_exact','0.5':'primary','1.0':'beta_100_exact'}
records={}
for b,c in case.items():
    with (F/c/'AU_EXTERNAL_INTERVAL_RESULTS.csv').open(encoding='utf-8-sig',newline='') as f:
        records[b]={(r['date'],int(r['interval_index'])):r for r in csv.DictReader(f) if r['control']=='ECF'}
sub=rest=0.;count=0;checks=0;maxerr=0.;hashes={}
for file in sorted((A/'trajectories').glob('*.npz')):
    date=file.stem;hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
    with np.load(file,allow_pickle=False) as v:
        available=v['available_mw'];capacity=v['capacity_mw'];raw=v['raw_request_mw']
        for b in case:
            ceil=v[f'beta_{b}_ceiling'];req=v[f'beta_{b}_request'];x=v[f'beta_{b}_award'];y=v[f'beta_{b}_delivery']
            assert np.array_equal(ceil[0],capacity)
            assert np.allclose(y,np.minimum(x,available),rtol=0,atol=1e-14)
            assert np.allclose(req,np.minimum(raw,ceil),rtol=0,atol=1e-14)
            for t in range(47):
                idle=x[t]-y[t]>2e-6
                target=np.where(idle & bool(int(records[b][date,t]['gate_triggered'])),y[t],capacity)
                nextceil=(1-float(b))*ceil[t]+float(b)*target
                maxerr=max(maxerr,float(np.max(np.abs(nextceil-ceil[t+1]))))
                assert np.allclose(nextceil,ceil[t+1],rtol=0,atol=1e-14)
                checks+=1
        delta=v['beta_1.0_delivery']-v['beta_0.5_delivery']
        mask=np.zeros_like(delta,dtype=bool)
        mask[1:]=(v['beta_1.0_ceiling'][1:]<v['beta_1.0_ceiling'][:-1]-1e-12)&(available[1:]>available[:-1]+1e-12)&(np.minimum(raw[1:],available[1:])>v['beta_1.0_ceiling'][1:]+1e-12)
        count+=int(mask.sum());sub+=float(delta[mask].sum())*.5;rest+=float(delta[~mask].sum())*.5
assert len(hashes)==80 and count==24453
assert abs(sub+rest-manifest['totals']['beta1_minus_beta05_MWh'])<1e-10
assert abs(sub+5.471252622560733)<1e-10
out={'days':80,'individual_updates_checked':checks*418,'max_update_error_MW':maxerr,'subgroup_connection_intervals':count,'subgroup_signed_delivery_difference_MWh':sub,'all_other_signed_delivery_difference_MWh':rest,'net_beta1_minus_beta05_MWh':sub+rest,'trajectory_sha256':hashes,'scope':'Same original recorded rho/g, independently checked equation evaluation and partition. No counterfactual removal of individual ceilings.'}
(A/'INDEPENDENT_TRAJECTORY_CHECK.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in out.items() if k!='trajectory_sha256'},indent=2))
