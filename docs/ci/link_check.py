import os, re, sys
root="docs"
bad=[]
link_pat=re.compile(r"\[(.*?)\]\((.*?)\)")
for dirpath,_,files in os.walk(root):
    for fn in files:
        if not fn.endswith(".md"): continue
        p=os.path.join(dirpath,fn)
        with open(p,encoding="utf-8") as fh:
            text = fh.read()
        for _, target in link_pat.findall(text):
            if target.startswith(("http://","https://","#")): continue
            tp=os.path.normpath(os.path.join(dirpath,target))
            if not os.path.exists(tp): bad.append((p,target))
if bad:
    for src,t in bad: print(f"Missing: {t} (from {src})")
    sys.exit(1)
