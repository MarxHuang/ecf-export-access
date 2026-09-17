"""Locate the power-accounting diagnostic residual without modifying the model."""
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
import run_mapping as run

A=Path(__file__).resolve().parent
p=run.verify_sources()
worst=json.loads((A/'POWER_ACCOUNTING_DIAGNOSTIC.json').read_text('utf-8'))['worst_record']
out=A/'days'/worst['mapping']/worst['date'];done=run.completed(out)
m=run.load_runner();connections=m.parse_dss_load_connections(Path(p['config']['dss_root']))
model=m.AustralianOpenDSSModel(p['config']['dss_root'],connections,load_power_factor=m.LOAD_POWER_FACTOR)
import opendssdirect as dss
with np.load(out/'VECTORS.npz') as z:
    k=5*worst['interval']+3;load=z['load_kw'][k];pv=z['pv_kw'][k];award=z['award_mw'][k]*1000;part=z['participant']
    screen,y,a=model.solve(load,pv,award,part)
    intended=np.maximum(load-np.where(part,pv,0),0)-y
    expected=dict(zip(model.load_names,intended));load_errors=[]
    groups=defaultdict(float);elements=[]
    for name in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(name)
        if not dss.CktElement.Enabled():continue
        powers=np.asarray(dss.CktElement.TotalPowers(),dtype=float)[0::2]
        total=float(powers.sum());kind=name.split('.')[0].lower();groups[kind]+=total
        elements.append(dict(name=name,terminal_p_kw=powers.tolist(),net_absorbed_kw=total))
        if kind=='load':
            short=name.split('.',1)[1].lower()
            load_errors.append(dict(name=name,actual_p_kw=total,setpoint_kw=expected[short],error_kw=total-expected[short]))
    report=dict(selected=worst,screen=screen.to_dict(),intended_connection_p_kw=float(intended.sum()),
                absorbed_p_by_element_class_kw=dict(groups),all_element_terminal_sum_kw=float(sum(groups.values())),
                largest_load_setpoint_errors=sorted(load_errors,key=lambda r:abs(r['error_kw']),reverse=True)[:10],
                nonstandard_elements=[e for e in elements if e['name'].split('.')[0].lower() not in ['line','transformer','load','vsource']],
                protocol_sha256=run.sha(A/'PROTOCOL.json'),controller_files=done['files'])
    (A/'POWER_RESIDUAL_DIAGNOSIS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['nonstandard_elements','largest_load_setpoint_errors']},indent=2))
