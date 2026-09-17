from pathlib import Path
import json
import numpy as np
A=Path(__file__).resolve().parent;T=A.parent/'164_CEILING_TRAJECTORY_REVIEW_20260907/trajectories'
rows=[]
for f in sorted(T.glob('*.npz')):
    with np.load(f) as z:
        x=z['beta_0.5_award'];y=z['beta_0.5_delivery'];a=z['available_mw']
        rows.append({'date':f.stem,'max_delivery_minus_award_MW':float((y-x).max()),
                     'max_delivery_minus_available_MW':float((y-a).max()),
                     'entries_delivery_above_award':int((y>x).sum()),
                     'max_identity_error_MW':float(np.abs(y-np.minimum(x,a)).max())})
out={'rows':rows,'max_delivery_minus_award_MW':max(r['max_delivery_minus_award_MW'] for r in rows),
     'max_identity_error_MW':max(r['max_identity_error_MW'] for r in rows)}
(A/'ROUNDING_INSPECTION.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print({k:v for k,v in out.items() if k!='rows'})
