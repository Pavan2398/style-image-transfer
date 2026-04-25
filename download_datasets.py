import os
import zipfile
from pathlib import Path
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_mscoco_subset(output_dir: str = './datasets/content'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / 'train2017.zip'

    if not zip_path.exists():
        print("Downloading MSCOCO train2017 subset (~2GB)...")
        print("URL: http://images.cocodataset.org/zips/train2017.zip")
        print("\nNote: This is a large download. Please manually download from:")
        print("https://cocodataset.org/#download")
        print(f"\nAfter downloading, extract to: {output_dir}")
        print(f"Then rename the extracted folder to 'content'")
        return False

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print("Extraction complete!")

    return True


def create_sample_datasets():
    content_dir = Path('./datasets/content')
    style_dir = Path('./datasets/style')

    content_dir.mkdir(parents=True, exist_ok=True)
    style_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("DATASET SETUP INSTRUCTIONS")
    print("=" * 60)

    print("""
CONTENT DATASET OPTIONS:

Option 1: Download MSCOCO (Recommended - 80K images)
  - Go to: https://cocodataset.org/#download
  - Download '2017 Train images [123K/5GB]'
  - Extract to ./datasets/content/

Option 2: Use a smaller subset
  - Download from: http://images.cocodataset.org/zips/train2017.zip
  - Extract first 5000 images only

Option 3: Use your own images
  - Create ./datasets/content/ folder
  - Add your content images (.jpg, .png)


STYLE DATASET OPTIONS:

  Create ./datasets/style/ folder and add:
  - Artistic paintings (Monet, Van Gogh, Picasso style)
  - Any image that defines the 'style' you want to transfer

  Example sources:
  - WikiArt dataset (Kaggle)
  - Custom artwork images
  - Pattern/texture images

Example folder structure:
  datasets/
    content/
      image001.jpg
      image002.jpg
      ...
    style/
      monet.jpg
      starry_night.jpg
      mosaic.jpg
      ...
""")

    print("=" * 60)

    if not any(content_dir.glob('*')):
        print("\nNo content images found. Please set up your datasets.")
        return False

    if not any(style_dir.glob('*')):
        print("\nNo style images found. Please add style images to ./datasets/style/")
        return False

    print("\nDatasets are ready!")
    return True


if __name__ == '__main__':
    create_sample_datasets()