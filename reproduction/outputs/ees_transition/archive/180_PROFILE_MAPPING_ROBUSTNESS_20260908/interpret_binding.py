"""Describe binding allocation without choosing or changing mapping scenarios."""
from pathlib import Path
import argparse,json
from collections import Counter
import numpy as np
import run_mapping as run

A=Path(__file__).resolve().parent

def main(partial):
    p=run.verify_sources();totals={};verified_days=0
    for mapping in p['mappings']:
        counts=Counter()
        for date in p['dates']:
            folder=A/'days'/mapping['id']/date
            if not (folder/'DONE.json').exists():continue
            run.completed(folder)
            with np.load(folder/'VECTORS.npz') as z:
                requests=z['request_mw'].reshape(48,5,-1)[:,2]
                awards=z['award_mw'].reshape(48,5,-1)[:,2]
                delivered=z['delivery_mw'].reshape(48,5,-1)[:,2]
                rp_awards=z['award_mw'].reshape(48,5,-1)[:,3]
                gates=z['gate'];idle=awards-delivered>2e-6
                for t in range(48):
                    total=float(requests[t].sum());unserved=float((requests[t]-awards[t]).sum())
                    # The 2 W figure is a reporting scale, not a new control threshold.
                    binding=bool(unserved>2e-6)
                    full=bool(np.max(np.abs(requests[t]-awards[t]))<1e-12)
                    if full:
                        released=float(np.maximum(rp_awards[t]-awards[t],0)[~idle[t]].sum())
                        assert released<2e-6 and not gates[t]
                    counts['intervals']+=1
                    counts['positive_request_intervals']+=total>2e-6
                    counts['request_reduction_above_2W']+=binding
                    counts['full_admitted_request_awarded']+=full
                    counts['unused_award_present']+=bool(idle[t].any())
                    counts['qualifying_intervals']+=bool(gates[t])
            verified_days+=1
        totals[mapping['id']]=dict(counts)
    complete=verified_days==400
    assert complete or partial,'Full descriptive result requires 400 completed days'
    report=dict(complete=complete,verified_days=verified_days,counts=totals,
                interpretation='Under a proportional allocation bounded by each admitted request, an original full award leaves no positive outside-access release in the replay. Binding is necessary, not sufficient for qualification.',
                protocol_sha256=run.sha(A/'PROTOCOL.json'),analysis_sha256=run.sha(Path(__file__)),control_changed=False)
    name='BINDING_INTERPRETATION.json' if complete else 'BINDING_PARTIAL.json'
    (A/name).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='counts'},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');args=ap.parse_args();main(args.partial)
