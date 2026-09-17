from pathlib import Path
import re,json,difflib,zipfile
import pymupdf
A=Path(__file__).resolve().parent;P=A.parents[1]/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
insertions={'main':[r'''
After a qualifying replay, unit gain sets each unused-award participant's
next ceiling to its completed delivery. The largest net losses relative to
gain 0.5 occur at connections whose output then recovers while the new ceiling
still limits their request (Section S10 of the \suppref{}).
'''],'supporting_information':[r'''unit gain, one interval with $g(t)=0$ restores every ceiling to $C_i$ at $t+1$.
At
''',r'''Individual ceiling trajectories locate the delivery differences in time.
They are reconstructed from the recorded allocation factors and replay
indicators, the original requests and available export, and the update in
\eqref{eq:supp_temporal_update}. Their interval totals agree with the original
records for all three gains. Relative to gain 0.5, unit gain loses a net
5.471~MWh over 24,453 connection--intervals where the preceding update tightened
the ceiling, available export then increased, and the ceiling lay below both
the current request and available export. The remaining connection--intervals
gain a net 1.036~MWh, leaving a total loss of 4.435~MWh over the same 80 days.
This partition locates the losses associated with output recovery after
tightening; it does not isolate the effect of removing one participant's
ceiling. Unit-gain ceilings return to capacity after any interval that does
not qualify, so the reversal does not require persistent restrictions after
the qualifying episode ends.

''']}
report={}
for root,items in insertions.items():
    old=(A/'before'/(root+'.tex')).read_text('utf-8');new=(P/(root+'.tex')).read_text('utf-8');stripped=new
    original=old
    for insert in items:
        assert stripped.count(insert)==1;stripped=stripped.replace(insert,'',1)
    if root=='supporting_information':
        boundary='has the highest utilization ratio, but it also has the lowest delivered export.\n\n'
        assert old.count(boundary+'\\clearpage\n')==1
        old=old.replace(boundary+'\\clearpage\n',boundary,1)
    assert stripped==old
    (A/(root+'.diff')).write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True))),encoding='utf-8')
    log=(A/('build_'+root)/(root+'.log')).read_text('utf-8',errors='replace')
    errors=[s for s in log.splitlines() if re.search(r'Overfull|(?:Citation|Reference).*undefined|undefined (?:references|citations|control sequence)|multiply defined|^!|not found',s,re.I)]
    assert not errors,errors
    doc=pymupdf.open(A/('build_'+root)/(root+'.pdf'))
    assert '??' not in ''.join(p.get_text() for p in doc)
    report[root]={'pages':len(doc),'only_expected_text_and_layout_changes':True,'blocking_findings':errors,'warnings':[s for s in log.splitlines() if 'Warning:' in s]}
assert (P/'references.bib').read_bytes()==(A/'before/references.bib').read_bytes()
with zipfile.ZipFile(A/'before/EES_OVERLEAF_CURRENT_20260907.zip') as z:
    figs=[n for n in z.namelist() if n.startswith('figures/')]
    assert len(figs)==11
    for f in figs:assert (P/f).read_bytes()==z.read(f)
report['existing_formulas_tables_labels_bib_figures_unchanged']=True
(A/'AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True,indent=2))
