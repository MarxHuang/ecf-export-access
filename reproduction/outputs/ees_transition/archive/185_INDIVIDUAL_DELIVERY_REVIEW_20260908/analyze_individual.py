from pathlib import Path
import hashlib,json
import numpy as np
A=Path(__file__).resolve().parent
F=A.parent/'151_FROZEN_SCENARIO_EVALUATION_20260907/runs/primary'
S=A.parent/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
R=A.parent/'164_CEILING_TRAJECTORY_REVIEW_20260907'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def near(a,b,tol=1e-10):
    err=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    assert np.isfinite(err) and err<tol,(err,tol)

def consecutive_days(flags,dates):
    run=np.zeros(flags.shape[1],dtype=int);longest=run.copy()
    for k,day in enumerate(dates):
        if k==0 or (day-dates[k-1]).astype(int)!=1:run[:]=0
        run=np.where(flags[k],run+1,0);longest=np.maximum(longest,run)
    return longest

def within_day_runs(flags):
    run=np.zeros(flags.shape[-1],dtype=int);longest=run.copy()
    for row in flags:run=np.where(row,run+1,0);longest=np.maximum(longest,run)
    return longest

def main():
    source_hashes={}
    def read(p):source_hashes[str(p)]=sha(p);return json.loads(p.read_text('utf-8'))
    proto=read(S/'PROTOCOL.json');old=read(S/'VERIFIED_COMPARISON.json')
    reco=read(R/'INDEPENDENT_TRAJECTORY_CHECK.json');primary=read(F/'AU_EXTERNAL_RUN_SUMMARY.json')
    assert old['protocol_sha256']==sha(S/'PROTOCOL.json')
    dates=proto['dates'];assert len(dates)==80 and len(set(dates))==80
    checkpoint={}
    for file in sorted((F/'day_checkpoints').glob('*.json')):
        data=json.loads(file.read_text('utf-8'))['result']
        if data['date'] in dates:
            assert data['date'] not in checkpoint
            checkpoint[data['date']]=data;source_hashes[str(file)]=sha(file)
    assert set(checkpoint)==set(dates)
    controls=['NO_FEEDBACK','MATCHED_UNIFORM','UNQUALIFIED_SAME_UPDATE']
    energies={c:[] for c in ['ECF',*controls]}
    downward=np.zeros(418,dtype=int);binding=np.zeros(418,dtype=int)
    longest_binding=np.zeros(418,dtype=int);binding_days=np.zeros(418,dtype=int)
    qualified_group=np.zeros(418,dtype=int);max_errors={}
    for date in dates:
        folder=S/'days'/date;done=read(folder/'DONE.json')
        assert done['protocol_sha256']==sha(S/'PROTOCOL.json')
        assert done['files']==old['output_hashes'][date]
        for name,h in done['files'].items():
            assert sha(folder/name)==h;source_hashes[str(folder/name)]=h
        reconstruction=R/'trajectories'/f'{date}.npz'
        assert sha(reconstruction)==reco['trajectory_sha256'][reconstruction.name]
        source_hashes[str(reconstruction)]=sha(reconstruction)
        original={(r['interval_index'],r['control']):r for r in checkpoint[date]['interval_records']}
        assert len(original)==192
        with np.load(folder/'TRAJECTORIES.npz') as z,np.load(reconstruction) as rz:
            raw=z['raw_request_mw'];cap=z['capacity_mw'];available=rz['available_mw']
            near(raw,rz['raw_request_mw']);near(cap,rz['capacity_mw'])
            assert raw.shape==available.shape==(48,418)
            for q in ['ceiling','request','award','delivery']:
                near(z['ECF_'+q],rz['beta_0.5_'+q])
            y={c:z[c+'_delivery'].copy() for c in ['ECF','UNQUALIFIED_SAME_UPDATE']}
            for c in controls[:2]:
                request=raw.copy()
                if c=='MATCHED_UNIFORM':request*=np.array([original[t,c]['uniform_factor'] for t in range(48)])[:,None]
                award=request*np.array([original[t,c]['request_scale'] for t in range(48)])[:,None]
                y[c]=np.minimum(award,available)
                for t in range(48):
                    row=original[t,c]
                    for key,values in [('admitted_request_kw',request),('authorized_export_kw',award),('delivered_export_kw',y[c])]:
                        err=abs(values[t].sum()*1000-row[key]);max_errors[key]=max(err,max_errors.get(key,0.))
                        assert err<1e-7,(date,t,c,key,err)
            for c,values in y.items():
                energies[c].append(values.sum(axis=0)*.5)
                if c!='UNQUALIFIED_SAME_UPDATE':
                    near(values.sum()*.5,sum(original[t,c]['delivered_export_kw'] for t in range(48))*.5/1000)
            near(y['ECF'],np.minimum(z['ECF_award'],available))
            near(y['UNQUALIFIED_SAME_UPDATE'],np.minimum(z['UNQUALIFIED_SAME_UPDATE_award'],available))
            before=z['ECF_ceiling'];after=.5*(before+z['ECF_target'])
            downward+=(after<before-1e-12).sum(axis=0)
            excluded=np.minimum(raw,cap)-z['ECF_request']>2e-6
            binding+=excluded.sum(axis=0);binding_days+=excluded.any(axis=0)
            longest_binding=np.maximum(longest_binding,within_day_runs(excluded))
            for t in range(48):
                if original[t,'ECF']['gate_triggered']:qualified_group+=z['ECF_idle'][t]
    energy={c:np.stack(rows) for c,rows in energies.items()}
    for c in ['ECF','NO_FEEDBACK','MATCHED_UNIFORM']:
        near(energy[c].sum(),primary['aggregate'][c]['delivered_export_mwh'],1e-8)
    near(energy['UNQUALIFIED_SAME_UPDATE'].sum(),old['totals']['UNQUALIFIED_SAME_UPDATE']['delivery'],1e-8)
    near((energy['ECF']-energy['UNQUALIFIED_SAME_UPDATE']).sum(axis=0),old['individual']['difference_MWh_by_connection_index'],1e-8)
    assert downward.sum()==old['counts']['ECF_tightening_updates']
    summaries={};members=[];all_daily=[];calendar=np.array(dates,dtype='datetime64[D]')
    for c in controls:
        diff=energy['ECF']-energy[c];net=diff.sum(axis=0)
        pos=net>1e-10;neg=net<-1e-10
        negative_days=(diff<-1e-10).sum(axis=0)
        run=consecutive_days(diff<-1e-10,calendar)
        near(net.sum(),energy['ECF'].sum()-energy[c].sum(),1e-8)
        near(net[pos].sum()+net[neg].sum()+net[~(pos|neg)].sum(),net.sum())
        qs=np.quantile(net*1000,[0,.05,.25,.5,.75,.95,1])
        summaries[c]=dict(positive_participants=int(pos.sum()),negative_participants=int(neg.sum()),unchanged_participants=int((~(pos|neg)).sum()),
            net_MWh=float(net.sum()),positive_subtotal_MWh=float(net[pos].sum()),negative_subtotal_MWh=float(net[neg].sum()),
            difference_kWh_quantiles=qs.tolist(),quantile_probabilities=[0,.05,.25,.5,.75,.95,1],
            participants_with_any_negative_day=int((negative_days>0).sum()),
            maximum_negative_days=int(negative_days.max()),maximum_consecutive_negative_days=int(run.max()),
            cumulative_losers_with_excluded_requests=int((neg & (binding>0)).sum()))
        for i in range(418):
            base=float(energy[c][:,i].sum())
            members.append(dict(participant_index=i,comparator=c,ECF_delivery_MWh=float(energy['ECF'][:,i].sum()),control_delivery_MWh=base,
                difference_kWh=float(net[i]*1000),relative_difference=float(net[i]/base) if base>1e-10 else None,
                positive_days=int((diff[:,i]>1e-10).sum()),negative_days=int(negative_days[i]),unchanged_days=int((abs(diff[:,i])<=1e-10).sum()),
                longest_consecutive_negative_days=int(run[i]),downward_updates=int(downward[i]),qualified_group_memberships=int(qualified_group[i]),
                ceiling_excluded_request_intervals=int(binding[i]),ceiling_excluded_request_days=int(binding_days[i]),
                longest_within_day_excluded_request_intervals=int(longest_binding[i])))
            for j,date in enumerate(dates):all_daily.append(dict(participant_index=i,date=date,comparator=c,difference_kWh=float(diff[j,i]*1000)))
    assert len(members)==1254 and len(all_daily)==100320
    for file,h in source_hashes.items():assert sha(Path(file))==h,file
    report=dict(days=80,participants=418,summaries=summaries,members=members,daily=all_daily,
        max_reconstruction_error_kw=max_errors,downward_updates=int(downward.sum()),
        ceiling_excluded_request_intervals=int(binding.sum()),new_power_flows=0,
        scope='Descriptive modeled individual delivery differences. Runs break across missing dates. No individual causal attribution, financial loss or participant adaptation is established.',
        source_hashes=source_hashes,analysis_sha256=sha(Path(__file__)),plan_sha256=sha(A/'PLAN.md'))
    (A/'INDIVIDUAL_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','members','daily']},indent=2))

if __name__=='__main__':main()
