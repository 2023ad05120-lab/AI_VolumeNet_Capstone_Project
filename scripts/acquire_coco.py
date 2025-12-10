#!/usr/bin/env python
# acquire_coco.py
# Downloads COCO dataset (or a subset) into the specified folder

import argparse
import os
import urllib.request
import zipfile

def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    url = "http://images.cocodataset.org/zips/train2017.zip"  # example subset
    zip_path = os.path.join(out_dir, "train2017.zip")

    print(f"Downloading COCO subset from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

    print(f"COCO dataset acquired at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    main(args.out)
