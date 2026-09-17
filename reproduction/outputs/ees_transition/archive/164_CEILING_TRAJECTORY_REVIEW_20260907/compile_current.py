"""Compile text-review PDFs with existing MiKTeX without latexmk/Perl."""
from pathlib import Path
import subprocess, os, json, sys
A=Path(__file__).resolve().parent
P=A.parents[1]/"EES_PARAGRAPH_REVIEW_MASTER_20260901"
BIN=Path(r"E:\Program Files\MikTeX\miktex\bin\x64")
root=sys.argv[1]
out=A/("build_"+root);out.mkdir(exist_ok=True)
env=os.environ.copy();env["BIBINPUTS"]=str(P)+os.pathsep+env.get("BIBINPUTS","");env["BSTINPUTS"]=str(P)+os.pathsep+env.get("BSTINPUTS","")
commands=[
([str(BIN/"pdflatex.exe"),"-interaction=nonstopmode","-halt-on-error","-file-line-error","-output-directory="+str(out),root+".tex"],P),
([str(BIN/"bibtex.exe"),root],out),
([str(BIN/"pdflatex.exe"),"-interaction=nonstopmode","-halt-on-error","-file-line-error","-output-directory="+str(out),root+".tex"],P),
([str(BIN/"pdflatex.exe"),"-interaction=nonstopmode","-halt-on-error","-file-line-error","-output-directory="+str(out),root+".tex"],P)]
for i,(cmd,cwd) in enumerate(commands):
    r=subprocess.run(cmd,cwd=cwd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=r.stdout.decode("utf-8",errors="replace")
    (out/f"step_{i}.txt").write_text(log,encoding="utf-8")
    print(json.dumps(dict(step=i,exit_code=r.returncode,tail=log[-1500:])),flush=True)
    if r.returncode:sys.exit(r.returncode)
print(str(out/(root+".pdf")))
