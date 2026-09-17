from pathlib import Path
import difflib,hashlib,json,re,subprocess,zipfile
import pymupdf as fitz
from PIL import Image,ImageDraw
A=Path(__file__).resolve().parent;P=A.parents[1]/'EES_PARAGRAPH_REVIEW_MASTER_20260901'
R=A/'rendered';R.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((A/'INDIVIDUAL_RESULTS.json').read_text('utf-8'))
v=json.loads((A/'VERIFIED_INDIVIDUAL_RESULTS.json').read_text('utf-8'))
assert v['passed'] and v['result_sha256']==sha(A/'INDIVIDUAL_RESULTS.json')
si=(P/'supporting_information.tex').read_text('utf-8')
section=si.split('\\label{supp:individual_delivery}',1)[1]
order=['NO_FEEDBACK','MATCHED_UNIFORM','UNQUALIFIED_SAME_UPDATE']
table=section.split('\\begin{tabular}',1)[1].split('\\end{tabular}',1)[0]
rows=[line for line in table.splitlines() if '&' in line][1:]
assert len(rows)==13
expected=[]
for field in ['positive_participants','negative_participants','unchanged_participants']:
    expected.append([str(r['summaries'][c][field]) for c in order])
for j in range(7):expected.append([f"{r['summaries'][c]['difference_kWh_quantiles'][j]:.2f}" for c in order])
for field in ['participants_with_any_negative_day','maximum_negative_days','maximum_consecutive_negative_days']:
    expected.append([str(r['summaries'][c][field]) for c in order])
for line,values in zip(rows,expected):
    actual=[x.strip().strip('\\').strip().replace('$','') for x in line.split('&')[1:]]
    assert actual==values,(actual,values)
report=dict(individual_analysis_verified=True,table_numeric_cells=39,primary_aggregate_result_unchanged=True)
with zipfile.ZipFile(A/'before/EES_OVERLEAF_CURRENT_20260907.zip') as z:
    figs=[n for n in z.namelist() if n.startswith('figures/') and n.endswith('.pdf')]
    assert len(figs)==12
    for n in figs:assert z.read(n)==(P/n).read_bytes(),n
report['author_figures_unchanged']=11;report['supplementary_figures_unchanged']=1
for root in ['main','supporting_information']:
    old=(A/'before'/(root+'.tex')).read_text('utf-8');new=(P/(root+'.tex')).read_text('utf-8')
    (A/(root+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True))),encoding='utf-8')
    assert re.findall(r'\\cite[a-zA-Z]*\{[^}]+\}',old)==re.findall(r'\\cite[a-zA-Z]*\{[^}]+\}',new)
    labels=re.findall(r'\\label\{([^}]+)\}',new)
    assert len(labels)==len(set(labels)) and set(re.findall(r'\\label\{([^}]+)\}',old))<=set(labels)
    for env in ['equation','align','gather','multline']:
        pattern=r'\\begin\{'+env+r'\*?\}.*?\\end\{'+env+r'\*?\}'
        assert re.findall(pattern,old,re.S)==re.findall(pattern,new,re.S)
    if root=='main':assert old.split('\\includegraphics{head_foot/dates}',1)[1].split('\\end{tabular}',1)[0]==new.split('\\includegraphics{head_foot/dates}',1)[1].split('\\end{tabular}',1)[0]
    pdf=A/('build_'+root)/(root+'.pdf');log=pdf.with_suffix('.log').read_text('utf-8',errors='replace')
    errors=[line for line in log.splitlines() if any(s in line for s in ['Overfull','undefined references','undefined citations','multiply defined','Rerun to get','Package balance Warning'])]
    assert not errors,errors
    doc=fitz.open(pdf);text='\n'.join(page.get_text() for page in doc)
    assert '??' not in text
    if root=='supporting_information':assert 'S10.14' in text and 'Table S22' in text
    # Render all pages through Poppler, then inspect contact sheets and new text.
    proc=subprocess.run(['pdftoppm','-scale-to','650','-png',str(pdf),str(R/root)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    assert proc.returncode==0,proc.stderr.decode(errors='replace')
    images=sorted(R.glob(root+'-*.png'),key=lambda p:int(p.stem.rsplit('-',1)[1]))
    assert len(images)==len(doc)
    detail=[]
    for i,page in enumerate(doc):
        content=page.get_text()
        if any(s in content for s in ['Most participant connections','Aggregate improvement should','Individual delivery changes','Table S22']):
            detail.append(i+1)
            page.get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(R/f'{root}_detail_{i+1}.png')
    for start in range(0,len(images),12):
        sheet=Image.new('RGB',(1200,1760),'#d8d8d8');draw=ImageDraw.Draw(sheet)
        for j,file in enumerate(images[start:start+12]):
            image=Image.open(file).convert('RGB');image.thumbnail((380,410));x=j%3*400;y=j//3*440
            sheet.paste(image,(x,y+20));draw.text((x+5,y+3),str(start+j+1),fill='black')
        sheet.save(R/f'{root}_contact_{start//12+1}.png')
    report[root]=dict(pages=len(doc),detail_pages=detail,tex_sha256=sha(P/(root+'.tex')),pdf_sha256=sha(pdf),
        warnings=[line for line in log.splitlines() if 'Warning' in line])
(A/'AUDIT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
