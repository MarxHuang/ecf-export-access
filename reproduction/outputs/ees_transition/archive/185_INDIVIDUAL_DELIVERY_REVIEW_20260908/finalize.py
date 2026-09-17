"""Promote verified PDFs and package the exact reviewed source files."""
from pathlib import Path
import hashlib, json, zipfile, shutil, re
A=Path(__file__).resolve().parent;P=A.parents[1]/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=json.loads((A/'AUDIT.json').read_text('utf-8'))
for root in ['main','supporting_information']:
 assert sha(P/(root+'.tex'))==audit[root]['tex_sha256']
 b=A/('build_'+root)/(root+'.pdf')
 assert sha(b)==audit[root]['pdf_sha256']
 shutil.copy2(b,P/(root+'.pdf'))
old=A/'before/EES_OVERLEAF_CURRENT_20260907.zip'
temp=A/'verified_source.zip'
allowed={'main.tex','supporting_information.tex','README.md','PUBLICATION_READINESS_LEDGER.md','SUBMISSION_TODO.md'}
added=['个体交付分布_修订与复核_20260908.md']
with zipfile.ZipFile(old) as before,zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as out:
 names=before.namelist();changed=[]
 for n in names:
  data=(P/n).read_bytes()
  if data!=before.read(n):assert n in allowed,n;changed.append(n)
  out.writestr(n,data)
 for n in added:
  assert n not in names
  out.write(P/n,n)
with zipfile.ZipFile(temp) as z:
 assert not z.testzip()
 names=z.namelist();assert len(names)==len(set(names))==37
 for n in names:assert z.read(n)==(P/n).read_bytes()
 for root in ['main.tex','supporting_information.tex','figure_data/mapping_tables.tex']:
  text=z.read(root).decode('utf-8')
  for kind,arg in re.findall(r'\\(includegraphics|input)(?:\[[^\]]*\])?\{([^}]+)\}',text):
   if '\\' in arg:raise AssertionError(('unresolved dynamic input',arg))
   candidates=[arg] if Path(arg).suffix else [arg+ext for ext in (['.tex'] if kind=='input' else ['.pdf','.png','.jpg','.eps'])]
   assert any(n in names for n in candidates),(root,arg)
 for n in names:assert not n.endswith(('.aux','.log','.out','.blg','.bbl','.toc','.fls','.fdb_latexmk'))
target=P/'EES_OVERLEAF_CURRENT_20260907.zip'
shutil.copy2(temp,target)
result=dict(source_entries=37,modified=changed,added=added,zip_sha256=sha(target),author_figures_unchanged=11,
 main_pages=audit['main']['pages'],si_pages=audit['supporting_information']['pages'],
 pdfs={root:sha(P/(root+'.pdf')) for root in ['main','supporting_information']},
 external_repository_published=False)
(A/'PACKAGE_CHECK.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=True))
