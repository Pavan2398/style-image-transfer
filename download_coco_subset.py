import os
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm
import shutil


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_first_n_images(zip_path: str, output_dir: str, n: int = 5000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting first {n} images from {zip_path}...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        image_files = [f for f in zip_ref.namelist() if f.endswith(('.jpg', '.jpeg', '.png'))]

        print(f"Total images in zip: {len(image_files)}")

        for i, img_name in enumerate(tqdm(image_files[:n], desc="Extracting")):
            try:
                zip_ref.extract(img_name, output_dir)
            except Exception as e:
                print(f"Error extracting {img_name}: {e}")

            if (i + 1) % 1000 == 0:
                print(f"Extracted {i + 1}/{n} images")

    print(f"\nDone! {n} images extracted to {output_dir}")


def download_and_extract_subset(
    output_dir: str = './datasets/content',
    num_images: int = 5000,
    use_val: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if use_val:
        zip_url = "http://images.cocodataset.org/zips/val2017.zip"
        zip_name = "val2017.zip"
    else:
        zip_url = "http://images.cocodataset.org/zips/train2017.zip"
        zip_name = "train2017.zip"

    zip_path = output_dir / zip_name

    if not zip_path.exists():
        print(f"Downloading MSCOCO 2017 {'validation' if use_val else 'train'} set...")
        print(f"URL: {zip_url}")
        print(f"This will download approximately {'1GB' if use_val else '18GB'}")

        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print("Download cancelled.")
            return False

        download_file(zip_url, str(zip_path))
    else:
        print(f"Found existing zip: {zip_path}")

    if use_val or num_images >= 118000:
        extract_first_n_images(str(zip_path), str(output_dir), num_images)
    else:
        extract_first_n_images(str(zip_path), str(output_dir), num_images)

    print("\n" + "=" * 60)
    print("CONTENT DATASET READY!")
    print(f"Location: {output_dir}")
    print("=" * 60)

    return True


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Download MSCOCO subset')
    parser.add_argument('--output', type=str, default='./datasets/content', help='Output directory')
    parser.add_argument('--num', type=int, default=5000, help='Number of images to extract')
    parser.add_argument('--use-val', action='store_true', help='Use validation set instead of train')

    args = parser.parse_args()

    download_and_extract_subset(args.output, args.num, args.use_val)