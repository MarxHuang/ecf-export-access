from pathlib import Path
import zipfile,hashlib,json,re
A=Path(__file__).resolve().parent;T=A.parents[1];P=T/"EES_PARAGRAPH_REVIEW_MASTER_20260901"
files=[P/n for n in ["main.tex","supporting_information.tex","references.bib","rsc.bst","README.md","ACTIVE_FIGURE_MANIFEST.md","SUBMISSION_TODO.md","修订与审阅循环_20260907.md","PUBLICATION_READINESS_LEDGER.md"]]
files+=sorted(p for p in (P/"head_foot").rglob("*") if p.is_file())
s=(P/"main.tex").read_text(encoding="utf-8")
figs=re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(figures/[^}]+)\}",s)
files +=[P/f for f in figs]
assert len(figs)==11 and len(files)==len(set(files))
out=P/"EES_OVERLEAF_CURRENT_20260907.zip"
manifest=[]
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        arc=p.relative_to(P).as_posix();z.write(p,arc)
        manifest.append({"file":arc,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()})
with zipfile.ZipFile(out) as z:
    assert z.testzip() is None
    for x in manifest: assert hashlib.sha256(z.read(x["file"])).hexdigest()==x["sha256"]
    assert not any(n.endswith((".aux",".log",".out",".toc",".bbl",".blg")) for n in z.namelist())
    assert z.read("main.tex")==(P/"main.tex").read_bytes()
for name in ["main","supporting_information"]:
    assert (P/(name+".pdf")).read_bytes()==(A/("build_"+name)/(name+".pdf")).read_bytes()
result={"archive":str(out),"entries":len(files),"archive_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"files":manifest,"compiled_PDFs_match":True}
(A/"FINAL_PACKAGE_CHECK.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="files"},ensure_ascii=False))
