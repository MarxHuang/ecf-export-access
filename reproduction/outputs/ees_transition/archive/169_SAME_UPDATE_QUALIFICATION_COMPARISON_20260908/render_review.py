from pathlib import Path
import subprocess,json
from PIL import Image,ImageDraw
import pymupdf
A=Path(__file__).resolve().parent
out=A/"rendered";out.mkdir(exist_ok=True)
index={}
for root in ["main","supporting_information"]:
    pdf=A/("build_"+root)/(root+".pdf")
    target=out/root;target.mkdir(exist_ok=True)
    r=subprocess.run(["pdftoppm","-r","70","-png",str(pdf),str(target/"page")],capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr.decode(errors="replace"))
    doc=pymupdf.open(pdf)
    pages=sorted(target.glob("page-*.png"))[:len(doc)]
    for start in range(0,len(pages),8):
        sheet=Image.new("RGB",(760,4*540),"#bbbbbb"); draw=ImageDraw.Draw(sheet)
        for k,p in enumerate(pages[start:start+8]):
            im=Image.open(p).convert("RGB");im.thumbnail((365,507))
            x=(k%2)*380+(380-im.width)//2;y=(k//2)*540+24
            sheet.paste(im,(x,y));draw.text(((k%2)*380+9,(k//2)*540+6),f"{root} p{start+k+1}",fill="black")
        sheet.save(out/f"{root}_contact_{start+1:02d}.png")
    doc=pymupdf.open(pdf)
    index[root]=[]
    for i,pg in enumerate(doc):
        terms=[term for term in ["same-update", "Without qualification", "6.42", "6.423", "component comparison", "ECF without replay", "S10.10"] if term in pg.get_text()]
        if terms or i==len(doc)-1:
            pg.get_pixmap(dpi=140).save(out/f'{root}_detail_{i+1:02d}.png')
            index[root].append({'page':i+1,'terms':terms})
(out/"PAGE_INDEX.json").write_text(json.dumps(index,indent=2),encoding="utf-8")
print(json.dumps(index,indent=2))
