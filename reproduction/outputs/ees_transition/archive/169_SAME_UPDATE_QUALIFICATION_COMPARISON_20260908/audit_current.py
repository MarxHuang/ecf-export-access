from pathlib import Path
import re,json,hashlib,difflib,collections
import pymupdf
A=Path(__file__).resolve().parent
P=A.parents[1]/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'documents':{}}
for root in ['main','supporting_information']:
    old=(A/'before'/f'{root}.tex').read_text('utf-8')
    new=(P/f'{root}.tex').read_text('utf-8')
    (A/f'{root}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))),encoding='utf-8')
    for pattern in [r'\\cite\{[^}]+\}',r'\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}']:
        assert re.findall(pattern,old)==re.findall(pattern,new)
    equations=lambda s:re.findall(r'\\begin\{(equation\*?|align\*?)\}(.*?)\\end\{\1\}',s,re.S)
    assert equations(old)==equations(new)
    labels=re.findall(r'\\label\{([^}]+)\}',new)
    assert len(labels)==len(set(labels))
    assert set(re.findall(r'\\label\{([^}]+)\}',old))<=set(labels)
    doc=pymupdf.open(A/f'build_{root}'/f'{root}.pdf')
    text='\n'.join(pg.get_text() for pg in doc)
    assert '??' not in text and '6.42' in text
    log=(A/f'build_{root}'/f'{root}.log').read_text(errors='replace')
    assert not re.search(r'(Reference|Citation).*undefined',log)
    assert 'Overfull' not in log and 'Label(s) may have changed' not in log
    result['documents'][root]={'pages':len(doc),'existing_equations_citations_figures_unchanged':True,'warnings':[l for l in log.splitlines() if 'Warning' in l]}
assert sha(P/'references.bib')==sha(A/'before/references.bib')
previous=json.loads((A.parent/'167_DISCOVERY_AND_GAIN_CLAIM_ALIGNMENT_20260908/FINAL_PACKAGE_CHECK.json').read_text('utf-8'))
figs=[f for f in previous['files'] if f['file'].startswith('figures/')]
assert len(figs)==11 and all(sha(P/f['file'])==f['sha256'] for f in figs)
v=json.loads((A/'VERIFIED_COMPARISON.json').read_text('utf-8'))
assert round(v['paired']['delivery']['sum_MWh'],2)==6.42
assert round(v['paired']['delivery']['sum_MWh']/v['totals']['UNQUALIFIED_SAME_UPDATE']['delivery']*100,2)==2.76
assert v['individual']['positive']==417 and v['individual']['negative']==1
assert not v['physical_failures']
result['eleven_figures_unchanged']=True
(A/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
