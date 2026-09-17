"""Check full participant/day outputs and run-length definitions independently."""
from pathlib import Path
from collections import defaultdict
import copy,hashlib,json,math
import numpy as np
A=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def verify(r):
    member={(v['comparator'],v['participant_index']):v for v in r['members']}
    assert len(member)==len(r['members'])==1254
    grouped=defaultdict(list)
    for v in r['daily']:grouped[v['comparator'],v['participant_index']].append(v)
    assert set(member)==set(grouped) and len(r['daily'])==100320
    for key,m in member.items():
        rows=sorted(grouped[key],key=lambda z:z['date'])
        assert len(rows)==80 and len({z['date'] for z in rows})==80
        values=[v['difference_kWh'] for v in rows]
        assert abs(math.fsum(values)-m['difference_kWh'])<1e-8
        assert abs(m['difference_kWh']-1000*(m['ECF_delivery_MWh']-m['control_delivery_MWh']))<1e-8
        neg=sum(v<-1e-7 for v in values);pos=sum(v>1e-7 for v in values)
        assert (pos,neg,80-pos-neg)==(m['positive_days'],m['negative_days'],m['unchanged_days'])
        previous=None;run=0;longest=0
        for z in rows:
            date=np.datetime64(z['date'],'D')
            if previous is None or int((date-previous)/np.timedelta64(1,'D'))!=1:run=0
            run=run+1 if z['difference_kWh']<-1e-7 else 0
            longest=max(longest,run);previous=date
        assert longest==m['longest_consecutive_negative_days']
    for c,s in r['summaries'].items():
        selected=[v for (cc,i),v in member.items() if cc==c]
        values=np.array([v['difference_kWh']/1000 for v in selected])
        assert len(values)==418
        positive=values>1e-10;negative=values<-1e-10
        assert int(positive.sum())==s['positive_participants']
        assert int(negative.sum())==s['negative_participants']
        assert 418-int(positive.sum())-int(negative.sum())==s['unchanged_participants']
        assert abs(math.fsum(values)-s['net_MWh'])<1e-8
        assert abs(math.fsum(values[positive])-s['positive_subtotal_MWh'])<1e-8
        assert abs(math.fsum(values[negative])-s['negative_subtotal_MWh'])<1e-8
        assert np.max(abs(np.quantile(values*1000,s['quantile_probabilities'])-s['difference_kWh_quantiles']))<1e-8
        assert max(v['negative_days'] for v in selected)==s['maximum_negative_days']
        assert max(v['longest_consecutive_negative_days'] for v in selected)==s['maximum_consecutive_negative_days']
    return True

def main():
    from analyze_individual import consecutive_days,within_day_runs
    dates=np.array(['2013-01-01','2013-01-02','2013-01-04','2013-01-05'],dtype='datetime64[D]')
    assert np.array_equal(consecutive_days(np.array([[1,0],[1,1],[1,1],[1,0]],dtype=bool),dates),[2,1])
    assert np.array_equal(within_day_runs(np.array([[1,0],[1,1],[0,1],[1,1]],dtype=bool)),[2,3])
    path=A/'INDIVIDUAL_RESULTS.json';before=sha(path);r=json.loads(path.read_text('utf-8'))
    assert r['analysis_sha256']==sha(A/'analyze_individual.py') and r['plan_sha256']==sha(A/'PLAN.md')
    assert verify(r)
    rejected=[]
    for name in ['net_energy','sign_count','run_length','daily_value','duplicate_date']:
        q=copy.deepcopy(r)
        if name=='net_energy':q['members'][0]['difference_kWh']+=1
        elif name=='sign_count':q['summaries']['NO_FEEDBACK']['positive_participants']+=1
        elif name=='run_length':q['members'][0]['longest_consecutive_negative_days']+=1
        elif name=='daily_value':q['daily'][0]['difference_kWh']+=1
        else:q['daily'][0]['date']=q['daily'][1]['date']
        try:verify(q)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Unrejected corruption '+name)
    for p,h in r['source_hashes'].items():assert sha(Path(p))==h,p
    assert sha(path)==before
    extrema={}
    for c in r['summaries']:
        members=[v for v in r['members'] if v['comparator']==c]
        extrema[c]=dict(largest_absolute_loss=min(members,key=lambda v:v['difference_kWh']),
            largest_relative_loss=min((v for v in members if v['relative_difference'] is not None),key=lambda v:v['relative_difference']))
    report=dict(passed=True,result_sha256=before,verifier_sha256=sha(Path(__file__)),
        checked_participant_contrasts=1254,checked_daily_contrasts=100320,
        rejected_faults=rejected,calendar_gap_and_interval_run_tests=True,source_hashes_unchanged=True,
        extrema=extrema,new_power_flows=0)
    (A/'VERIFIED_INDIVIDUAL_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='extrema'},indent=2))

if __name__=='__main__':main()
