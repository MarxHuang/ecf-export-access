"""Check that independent validation rejects corrupted in-memory trace fields.

Scientific files and the running experiment are never modified. Each fault is
injected only into arrays returned to this process after the file hash check.
"""
from pathlib import Path
import importlib.util,json
import numpy as np
A=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('independent_checker',A/'verify_results.py')
checker=importlib.util.module_from_spec(s);s.loader.exec_module(checker)
original=np.load
target=A/'jobs/block09_ECF_0.25/2013-05-17/TRAJECTORIES.npz'
assert (target.parent/'DONE.json').exists(),'Wait for a real completed cross-day checkpoint'
with original(target) as z: participant=int(np.flatnonzero(z['participant'])[0])
faults=[('start_ceiling_full_mw',participant),('end_ceiling_full_mw',participant),('ceiling',(1,0)),('target',(0,0)),('delivery',(0,0)),('uniform_request',(0,0)),('replay_delivery',(0,0))]
class Altered:
    def __init__(self,values):self.values=values
    def __getitem__(self,key):return self.values[key]
    def __enter__(self):return self
    def __exit__(self,*args):return False
results=[]
for field,index in faults:
    injected=[False]
    def faulty(path,*args,**kwargs):
        if Path(path).resolve()!=target.resolve():return original(path,*args,**kwargs)
        with original(path,*args,**kwargs) as z:values={k:z[k].copy() for k in z.files}
        values[field][index]+=1e-4;injected[0]=True
        return Altered(values)
    np.load=faulty
    rejected=False
    try:checker.main(True)
    except AssertionError:rejected=True
    finally:np.load=original
    assert injected[0] and rejected,(field,'validation did not reject the injected fault')
    results.append({'field':field,'magnitude_MW':1e-4,'rejected':True})
(A/'VERIFIER_NEGATIVE_TESTS.json').write_text(json.dumps({'method':'In-memory fault injection; disk files unchanged','tests':results},indent=2),encoding='utf-8')
print('Rejected all',len(results),'injected trace faults; scientific files unchanged')
