"""Capture the stopped smoke input without changing the frozen controller."""
from pathlib import Path
import json
import numpy as np
import run_feedback as run
A=Path(__file__).resolve().parent
original=run.choose_transition
def capture(**kwargs):
    try:return original(**kwargs)
    except ValueError as e:
        if 'conflicts with metered positive import' not in str(e):raise
        data={k:v for k,v in kwargs.items() if isinstance(v,np.ndarray)}
        np.savez_compressed(A/'CONFLICT_INPUT.npz',**data)
        import control_step as ctl
        origscreen=ctl.screen_record_delivery
        def inner(**kw):
            conflict=(kw['replay_delivery_mw']*1000>1e-9)&(kw['net_import_kw']>1e-9)&kw['participant']
            ids=np.flatnonzero(conflict)
            rows=[{'connection_index':int(i),'award_W':kwargs['allocation'][i]*1e6,
                'delivery_W':kwargs['delivery'][i]*1e6,'request_W':kwargs['request'][i]*1e6,
                'record_W':kwargs['record'][i]*1e6,'candidate_replay_W':kw['replay_delivery_mw'][i]*1e6,
                'metered_import_W':kw['net_import_kw'][i]*1000} for i in ids]
            (A/'CONFLICT_INSPECTION.json').write_text(json.dumps({'error':str(e),'connections':rows},indent=2),encoding='utf-8')
            print(json.dumps(rows,indent=2));return origscreen(**kw)
        ctl.screen_record_delivery=inner
        try:original(**kwargs)
        finally:ctl.screen_record_delivery=origscreen
run.choose_transition=capture
p=run.verify();case=next(c for c in p['cases'] if c['id']=='R10_POS_POINT')
try:run.task(case,True)
except ValueError as e:print('Preserved stopped input:',e)
