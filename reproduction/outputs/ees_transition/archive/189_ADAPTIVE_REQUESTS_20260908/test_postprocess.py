"""Synthetic paired contrasts test totals, signs, missing cells and seed nesting."""
import copy
from datetime import date,timedelta
import numpy as np
from prepare_publication_data import summarize,day_weights,describe,A,sha,write,MODES,SEEDS,CONTROLS

def main():
    dates=[str(date(2020,1,1)+timedelta(days=i)) for i in range(10)]
    rows=[]
    for mode in MODES:
        for si,seed in enumerate(SEEDS):
            for di,d in enumerate(dates):
                for control in CONTROLS:
                    gain=(si+1)*(.02 if mode=='ADAPTIVE' else .01) if control=='ECF' else 0
                    y=1+di*.1+gain
                    rows.append(dict(mode=mode,seed=seed,date=d,control=control,delivery_mwh=y,authorization_mwh=y+.4,idle_mwh=.4,source_exchange_mwh=-2+gain))
    r=summarize(rows,dates,[dates[:4],dates[4:]])
    primary=[q for q in r['ecf_contrasts'] if q['seed']=='SEED_MEAN']
    for q in primary:
        expected=.4 if q['mode']=='ADAPTIVE' else .2
        assert abs(q['total_MWh']-expected)<1e-12
        assert abs(q['mean_daily_kWh']-expected*100)<1e-10
        assert abs(q['CI95_low_daily_kWh']-q['CI95_high_daily_kWh'])<1e-10
        assert q['positive_dates']==10 and q['negative_dates']==0
    for q in r['change_in_ecf_contrast_due_to_preference_updates']:
        if q['seed']=='SEED_MEAN':assert abs(q['total_MWh']-.2)<1e-12
    for q in r['ecf_contrasts']:
        blocks=[b for b in r['block_contrasts'] if all(b[k]==q[k] for k in ['mode','seed','control'])]
        assert abs(sum(b['total_MWh'] for b in blocks)-q['total_MWh'])<1e-12
    w=day_weights(dates);assert w.shape==(10000,10) and np.array_equal(w[:5000],-w[5000:])
    mixed=describe(np.array([-1,1]*5,dtype=float),w)
    assert mixed['positive_dates']==mixed['negative_dates']==5 and mixed['CI95_low_daily_kWh']<0<mixed['CI95_high_daily_kWh']
    rejected=[]
    for label,data in [('missing_date_control',rows[:-1]),('duplicate_date_control',rows+[rows[0]])]:
        try:summarize(data,dates,[dates])
        except AssertionError:rejected.append(label)
        else:raise AssertionError(label+' was accepted')
    invalid=copy.deepcopy(rows);invalid[0]['delivery_mwh']+=1
    try:summarize(invalid,dates,[dates])
    except AssertionError:rejected.append('broken_energy_identity')
    else:raise AssertionError('Broken identity was accepted')
    # Physical outcome flags are never filters in summarize: daily totals must stay.
    marked=copy.deepcopy(rows)
    for q in marked:q['failed_observed_intervals']=48
    rr=summarize(marked,dates,[dates[:4],dates[4:]])
    assert rr==r
    write(A/'POSTPROCESS_TESTS.json',{'passed':True,'synthetic_input_rows':len(rows),'seed_mean_is_not_seed_sum':True,'paired_mode_contrast_verified':True,'block_totals_add':True,'physical_flags_do_not_filter_totals':True,'negative_tests':rejected,'producer_sha256':sha(A/'prepare_publication_data.py'),'test_sha256':sha(A/'test_postprocess.py')})
    print({'passed':True,'rows':len(rows),'rejected':rejected})

if __name__=='__main__':main()
