import csv,json
from pathlib import Path
from prepare_mapping import mapping_check,A,B

p=json.loads((B/'PROTOCOL.json').read_text('utf-8'))
with Path(p['config']['mapping_file']).open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
mapping_check(rows,rows)
changed=[dict(r) for r in rows]
group=[i for i,r in enumerate(rows) if r['participant_33pct']=='1']
i,j=group[:2]
changed[i]['ausgrid_customer_id'],changed[j]['ausgrid_customer_id']=changed[j]['ausgrid_customer_id'],changed[i]['ausgrid_customer_id']
mapping_check(rows,changed)
rejected=[]
for kind in ['customer_not_in_source','changed_phase','duplicate_load','changed_participation','lost_row']:
    corrupt=[dict(r) for r in rows]
    if kind=='customer_not_in_source':corrupt[0]['ausgrid_customer_id']='99999'
    if kind=='changed_phase':corrupt[0]['phase']='9'
    if kind=='duplicate_load':corrupt[1]['load_name']=corrupt[0]['load_name']
    if kind=='changed_participation':corrupt[0]['participant_33pct']='1' if corrupt[0]['participant_33pct']=='0' else '0'
    if kind=='lost_row':corrupt.pop()
    try:mapping_check(rows,corrupt)
    except AssertionError:rejected.append(kind)
    else:raise RuntimeError(f'Accepted invalid map: {kind}')
report=dict(valid_within_group_exchange=True,rejected=rejected)
(A/'MAPPING_UNIT_TESTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
