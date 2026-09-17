"""Independent active-power balance diagnostics on completed replay AC captures."""
from pathlib import Path
import json
import numpy as np
import run_mapping as run

A=Path(__file__).resolve().parent
p=run.verify_sources()
rows=[]
for mapping in p['mappings']:
    for date in p['dates']:
        out=A/'days'/mapping['id']/date
        if not (out/'REPLAY_AC.json').exists():continue
        done=run.completed(out)
        ac=json.loads((out/'REPLAY_AC.json').read_text('utf-8'))
        assert ac['controller_files']==done['files']
        with np.load(out/'VECTORS.npz') as z:
            part=z['participant']
            for t,s in enumerate(ac['screens']):
                k=t*5+3;load=z['load_kw'][k];pv=np.where(part,z['pv_kw'][k],0)
                award=z['award_mw'][k]*1000;delivery=z['delivery_mw'][k]*1000
                powers={'load_only':load,'authorization':np.where(part,-award,load),
                        'delivery':np.maximum(load-pv,0)-delivery}
                for stage,connection in powers.items():
                    screen=s[stage]
                    # OpenDSS TotalPower is negative for feeder supply.
                    residual=float(connection.sum()+screen['circuit_loss_kw']+screen['source_terminal_p_kw'])
                    rows.append(dict(mapping=mapping['id'],date=date,interval=t,stage=stage,
                                     residual_kW=residual,scale_kW=float(np.abs(connection).sum()),
                                     converged=screen['converged']))
assert rows,'No completed replay AC captures'
maximum=max(rows,key=lambda r:abs(r['residual_kW']))
report=dict(scope='Active power only; reactive shunts and native line ampacity are not inferred.',
            records=len(rows),mapping_days=len(rows)//144,max_absolute_residual_kW=abs(maximum['residual_kW']),
            worst_record=maximum,protocol_sha256=run.sha(A/'PROTOCOL.json'),diagnostic_sha256=run.sha(Path(__file__)))
(A/'POWER_ACCOUNTING_DIAGNOSTIC.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
