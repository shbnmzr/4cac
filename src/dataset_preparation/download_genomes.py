#!/usr/bin/env python3
"""
Download and organize GenBank genomes into train/val/test splits based on release dates,
with the following sampling strategy: first sample one genome per unique taxid, then top up by randomly
selected from the remaining, unselected genomes, if needed.

Usage:
    python prepare_datasets.py \
        --category archaea \
        --train_cutoff 2021-12-31 \
        --val_cutoff 2022-04-01 \
        --test_end 2023-04-01 \
        --base_dir ./my_dataset \
        --max_train 27000 \
        --max_val 2000 \
        --max_test 2129 \
        --random_seed 123
"""

import argparse
import gzip
import logging
import shutil
import time
import requests
import random
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from Bio import SeqIO

# -----------------------------
# CONFIGURATION
# -----------------------------
MAX_RETRIES = 3
RETRY_DELAY = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# -----------------------------
# HELPERS
# -----------------------------
def ensure_directories(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

def download_text_file(url: str, dest: Path) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info(f"Downloading: {url}")
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            dest.write_text(response.text)
            return
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed for {url}: {e}")
            time.sleep(RETRY_DELAY)
    raise RuntimeError(f"Failed to download {url} after {MAX_RETRIES} attempts.")

def smart_downsample(entries: List[Tuple[str, datetime, str]], limit: int) -> List[Tuple[str, datetime]]:
    taxid_map = {}
    extra_pool = []
    for entry in entries:
        path, date, taxid = entry
        taxid_map.setdefault(taxid, []).append((path, date))

    # One genome per unique taxid
    unique_reps = [random.choice(paths) for paths in taxid_map.values()]
    used_paths = set(p for p, _ in unique_reps)

    # Gather the remaining ones
    for paths in taxid_map.values():
        for p in paths:
            if p not in used_paths:
                extra_pool.append(p)

    needed = max(0, limit - len(unique_reps))
    if needed > 0:
        extra_sample = random.sample(extra_pool, min(needed, len(extra_pool)))
        return unique_reps + extra_sample
    return random.sample(unique_reps, limit) if len(unique_reps) > limit else unique_reps

def parse_ftp_paths(summary_file: Path, train_cutoff: datetime, val_cutoff: datetime, test_end: datetime, max_train: int, max_val: int, max_test: int) -> dict:
    with summary_file.open() as f:
        header = next(line for line in f if line.startswith("#assembly_accession")).strip().split("\t")

    ftp_col_index = header.index("ftp_path")
    date_col_index = header.index("seq_rel_date")
    taxid_index = header.index("taxid")

    raw_entries = {"train": [], "val": [], "test": []}
    with summary_file.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) > 14 and parts[11] in {"Complete Genome", "Chromosome", "Scaffold"} and parts[10] == "latest":
                try:
                    release_date = datetime.strptime(parts[date_col_index], "%Y-%m-%d")
                    path = parts[ftp_col_index]
                    taxid = parts[taxid_index]
                    if release_date < train_cutoff:
                        raw_entries["train"].append((path, release_date, taxid))
                    elif release_date < val_cutoff:
                        raw_entries["val"].append((path, release_date, taxid))
                    elif release_date < test_end:
                        raw_entries["test"].append((path, release_date, taxid))
                except ValueError:
                    continue

    return {
        "train": smart_downsample(raw_entries["train"], max_train),
        "val": smart_downsample(raw_entries["val"], max_val),
        "test": smart_downsample(raw_entries["test"], max_test),
    }

def save_list_to_file(data: List[str], dest: Path) -> None:
    dest.write_text("\n".join(data))

def is_valid_fasta(path: Path) -> bool:
    try:
        with open(path) as f:
            return any(line.startswith(">") for line in f)
    except Exception:
        return False

def has_valid_contig(path: Path, min_len=1000) -> bool:
    try:
        return any(len(rec.seq) >= min_len for rec in SeqIO.parse(path, "fasta"))
    except Exception:
        return False

def download_file(url: str, dest_path: Path) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            temp_path = dest_path.with_suffix(".tmp")
            with requests.get(url.replace("ftp://", "https://"), stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            shutil.move(temp_path, dest_path)
            return True
        except Exception as e:
            logging.warning(f"Attempt {attempt} failed for {url}: {e}")
            time.sleep(RETRY_DELAY)
    logging.error(f"Failed to download {url} after {MAX_RETRIES} attempts.")
    return False

def decompress_gz(gz_file: Path) -> bool:
    try:
        output_path = gz_file.with_suffix('')
        with gzip.open(gz_file, 'rb') as f_in, open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        gz_file.unlink()
        if output_path.stat().st_size < 500 or not is_valid_fasta(output_path) or not has_valid_contig(output_path):
            logging.warning(f"Removing invalid FASTA: {output_path.name}")
            output_path.unlink()
            return False
        return True
    except Exception as e:
        logging.warning(f"Failed to decompress {gz_file.name}: {e}")
        return False

def download_split(split_name: str, entries: List[Tuple[str, datetime]], output_dir: Path, metadata_dir: Path, category: str):
    logging.info(f"Downloading {split_name.upper()} genomes...")
    failed_urls = []
    ensure_directories(output_dir)

    for path, _ in entries:
        accession = path.split("/")[-1]
        url = f"{path}/{accession}_genomic.fna.gz"
        dest_path = output_dir / url.split("/")[-1]

        if dest_path.exists():
            continue

        success = download_file(url, dest_path)
        if success:
            if not decompress_gz(dest_path):
                failed_urls.append(url)
        else:
            failed_urls.append(url)

    if failed_urls:
        save_list_to_file(failed_urls, metadata_dir / f"{category}_{split_name}_failed.txt")

def main(category: str, base_dir: Path, train_cutoff: datetime, val_cutoff: datetime, test_end: datetime, max_train: int, max_val: int, max_test: int, random_seed: int = None):
    if random_seed is not None:
        random.seed(random_seed)

    metadata_dir = base_dir / "metadata"
    train_dir = base_dir / "train" / category
    val_dir = base_dir / "val" / category
    test_dir = base_dir / "test" / category

    ensure_directories(train_dir, val_dir, test_dir, metadata_dir)

    summary_file = metadata_dir / f"{category}_assembly_summary.txt"
    url = f"https://ftp.ncbi.nlm.nih.gov/genomes/genbank/{category}/assembly_summary.txt"
    download_text_file(url, summary_file)

    dataset_entries = parse_ftp_paths(summary_file, train_cutoff, val_cutoff, test_end, max_train, max_val, max_test)
    for split in ["train", "val", "test"]:
        logging.info(f"{split.upper()}: {len(dataset_entries[split])} genomes")
        save_list_to_file(
            [p for p, _ in dataset_entries[split]],
            metadata_dir / f"{category}_{split}_ftp_paths.txt"
        )

    download_split("train", dataset_entries["train"], train_dir, metadata_dir, category)
    download_split("val", dataset_entries["val"], val_dir, metadata_dir, category)
    download_split("test", dataset_entries["test"], test_dir, metadata_dir, category)

    logging.info(f"Completed downloading and organizing datasets for category: {category}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GenBank genomes into train/val/test splits.")
    parser.add_argument("--category", required=True, help="Category (e.g., archaea, bacteria, viral, plasmid, fungi, protozoa)")
    parser.add_argument("--base_dir", default="./4CAC_dataset", help="Base directory for storing data")
    parser.add_argument("--train_cutoff", required=True, help="Date before which genomes go to TRAIN (YYYY-MM-DD)")
    parser.add_argument("--val_cutoff", required=True, help="Date before which genomes go to VAL (YYYY-MM-DD)")
    parser.add_argument("--test_end", required=True, help="Date before which genomes go to TEST (YYYY-MM-DD)")
    parser.add_argument("--max_train", type=int, default=None, help="Maximum number of genomes for training split")
    parser.add_argument("--max_val", type=int, default=None, help="Maximum number of genomes for validation split")
    parser.add_argument("--max_test", type=int, default=None, help="Maximum number of genomes for test split")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducible sampling")
    args = parser.parse_args()

    main(
        args.category,
        Path(args.base_dir),
        datetime.strptime(args.train_cutoff, "%Y-%m-%d"),
        datetime.strptime(args.val_cutoff, "%Y-%m-%d"),
        datetime.strptime(args.test_end, "%Y-%m-%d"),
        args.max_train,
        args.max_val,
        args.max_test,
        args.random_seed
    )
