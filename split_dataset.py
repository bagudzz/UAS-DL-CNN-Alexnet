import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def list_images(folder: Path):
    files = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files

def safe_copy(src: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    # avoid overwrite if same name exists
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{abs(hash(str(src)))}{src.suffix}"
    shutil.copy2(src, dst)

def main():
    # UBAH INI sesuai lokasi dataset mentah kamu
    raw_root = Path("data_raw")   # misal: data_raw/cat dan data_raw/dog
    out_root = Path("data")       # akan membuat data/train dan data/test
    test_size = 0.2
    seed = 42

    cat_dir = raw_root / "cat"
    dog_dir = raw_root / "dog"

    if not cat_dir.exists() or not dog_dir.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan. Pastikan ada:\n{cat_dir}\n{dog_dir}"
        )

    cat_imgs = list_images(cat_dir)
    dog_imgs = list_images(dog_dir)

    if len(cat_imgs) == 0 or len(dog_imgs) == 0:
        raise RuntimeError("Tidak ada gambar yang terbaca di folder cat/dog.")

    all_imgs = cat_imgs + dog_imgs
    train_imgs, test_imgs = train_test_split(all_imgs, test_size=test_size, random_state=seed, shuffle=True)

    # copy train
    for p in train_imgs:
        label = p.parent.name  # cat/dog
        safe_copy(p, out_root / "train" / label)

    # copy test
    for p in test_imgs:
        label = p.parent.name
        safe_copy(p, out_root / "test" / label)

    print("Selesai split.")
    print("Train:", len(train_imgs))
    print("Test :", len(test_imgs))
    print("Output folder:", out_root.resolve())

if __name__ == "__main__":
    main()
