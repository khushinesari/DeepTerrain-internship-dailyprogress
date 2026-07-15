
"""dataset_stats.py"""
from pathlib import Path
from collections import Counter
CLASS_NAMES={0:"person",1:"vehicle",2:"animal"}
def dataset_statistics(root):
    root=Path(root)
    for split in ["train","val","test"]:
        imgs=list((root/split/"images").glob("*"))
        lbls=list((root/split/"labels").glob("*.txt"))
        c=Counter(); obj=neg=inv=0
        for lbl in lbls:
            if lbl.stat().st_size==0:
                neg+=1; continue
            for line in open(lbl):
                p=line.split()
                if len(p)!=5:
                    inv+=1; continue
                c[int(p[0])]+=1; obj+=1
        print(f"\n{split.upper()}")
        print("Images:",len(imgs))
        print("Labels:",len(lbls))
        print("Objects:",obj)
        print("Negative:",neg)
        print("Invalid:",inv)
        for k,v in CLASS_NAMES.items():
            print(v,c.get(k,0))
