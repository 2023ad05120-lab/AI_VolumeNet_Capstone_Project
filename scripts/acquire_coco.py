#!/usr/bin/env python
# Acquire COCO dataset (download or placeholder)

import argparse, os, urllib.request, zipfile

def main(out_dir, download=False):
    os.makedirs(out_dir, exist_ok=True)
    if download:
        url = "http://images.cocodataset.org/zips/train2017.zip"
        zip_path = os.path.join(out_dir, "train2017.zip")
        print(f"[COCO] Downloading subset from {url}...")
        urllib.request.urlretrieve(url, zip_path)
        print("[COCO] Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)
        print(f"[COCO] Dataset acquired at {out_dir}")
    else:
        print(f"[COCO] Placeholder created at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--download", action="store_true", help="Download actual COCO subset")
    args = parser.parse_args()
    main(args.out, args.download)
