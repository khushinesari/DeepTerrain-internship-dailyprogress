import hashlib
import shutil
from pathlib import Path

from config import VALID_IMAGE_EXTENSIONS


def ensure_dir(folder: Path):
    folder.mkdir(parents=True, exist_ok=True)


def get_images(folder: Path):

    images = []

    for ext in VALID_IMAGE_EXTENSIONS:
        images.extend(folder.glob(f"*{ext}"))

    return sorted(images)


def copy_file(src: Path, dst: Path):

    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def md5(file_path):

    hash_md5 = hashlib.md5()

    with open(file_path, "rb") as f:

        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def duplicate_check(image_paths):

    hashes = {}

    duplicates = []

    for img in image_paths:

        h = md5(img)

        if h in hashes:
            duplicates.append(img)
        else:
            hashes[h] = img

    return duplicates
