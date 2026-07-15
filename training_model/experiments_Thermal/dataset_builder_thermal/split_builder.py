
"""split_builder.py"""
import random, shutil
from pathlib import Path
from config import MERGED_OUTPUT, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
from utils import ensure_dir
VALID_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
def copy_pair(image,destination):
    img_dst=destination/"images"; lbl_dst=destination/"labels"
    ensure_dir(img_dst); ensure_dir(lbl_dst)
    label=MERGED_OUTPUT/"labels"/(image.stem+".txt")
    if not label.exists():
        print(f"[WARNING] Missing label: {label.name}"); return False
    shutil.copy2(image,img_dst/image.name); shutil.copy2(label,lbl_dst/label.name); return True
def split_dataset():
    assert abs(TRAIN_RATIO+VAL_RATIO+TEST_RATIO-1.0)<1e-6
    imgs=sorted(p for p in (MERGED_OUTPUT/"images").iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
    random.seed(RANDOM_SEED); random.shuffle(imgs)
    n=len(imgs); a=int(n*TRAIN_RATIO); b=a+int(n*VAL_RATIO)
    splits={"train":imgs[:a],"val":imgs[a:b],"test":imgs[b:]}
    copied=0
    for s,v in splits.items():
        d=MERGED_OUTPUT/s; ensure_dir(d/"images"); ensure_dir(d/"labels")
        for im in v: copied+=1 if copy_pair(im,d) else 0
    print("Split complete"); print({k:len(v) for k,v in splits.items()})
