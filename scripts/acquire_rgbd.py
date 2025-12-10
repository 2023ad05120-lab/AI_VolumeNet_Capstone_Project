#!/usr/bin/env python
# Acquire RGB-D dataset (download or placeholder)

import argparse, os

def main(out_dir, download=False):
    os.makedirs(out_dir, exist_ok=True)
    if download:
        # TODO: Add actual RGB-D dataset download logic (e.g., NYU Depth V2, SUN RGB-D)
        print(f"[RGB-D] Download logic not yet implemented. Placeholder only.")
    else:
        print(f"[RGB-D] Placeholder created at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--download", action="store_true", help="Download actual RGB-D dataset")
    args = parser.parse_args()
    main(args.out, args.download)
