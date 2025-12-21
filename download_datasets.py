#!/usr/bin/env python3
"""
Download datasets for audio watermarking experiments.

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset ljspeech
    python download_datasets.py --list
"""

import os
import argparse
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm

DATA_ROOT = Path("./datasets")
DATA_ROOT.mkdir(exist_ok=True)

DATASETS = {
    # === РЕЧЬ (Speech) ===
    'ljspeech': {
        'url': 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
        'archive': 'LJSpeech-1.1.tar.bz2',
        'extract_dir': 'LJSpeech-1.1',
        'description': 'LJSpeech - Single speaker English speech',
        'compression': 'bz2',
        'category': 'speech'
    },
    'librispeech-dev': {
        'url': 'https://www.openslr.org/resources/12/dev-clean.tar.gz',
        'archive': 'dev-clean.tar.gz',
        'extract_dir': 'LibriSpeech',
        'description': 'LibriSpeech dev-clean - English speech',
        'compression': 'gz',
        'category': 'speech'
    },
    'librispeech-test': {
        'url': 'https://www.openslr.org/resources/12/test-clean.tar.gz',
        'archive': 'test-clean.tar.gz',
        'extract_dir': 'LibriSpeech',
        'description': 'LibriSpeech test-clean - English speech',
        'compression': 'gz',
        'category': 'speech'
    },
    'daps': {
        'url': 'https://zenodo.org/record/4660670/files/daps.tar.gz?download=1',
        'archive': 'daps.tar.gz',
        'extract_dir': 'daps',
        'description': 'DAPS - Device and Produced Speech',
        'compression': 'gz',
        'category': 'speech'
    },
    
    # === МУЗЫКА (Music) ===
    'gtzan': {
        'url': 'https://huggingface.co/datasets/marsyas/gtzan/resolve/main/data/genres.tar.gz',
        'archive': 'gtzan_genres.tar.gz',
        'extract_dir': 'gtzan_genres',
        'description': 'GTZAN - 10 жанров музыки: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock',
        'compression': 'gz',
        'category': 'music'
    },
    'fma-small': {
        'url': 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',
        'archive': 'fma_small.zip',
        'extract_dir': 'fma_small',
        'description': 'FMA Small - 8000 треков, 8 жанров',
        'compression': 'zip',
        'category': 'music'
    },
    
    # === ТЕХНИЧЕСКИЕ И ОКРУЖАЮЩИЕ ЗВУКИ (Environmental/Technical sounds) ===
    'esc50': {
        'url': 'https://github.com/karoldvl/ESC-50/archive/master.zip',
        'archive': 'ESC-50-master.zip',
        'extract_dir': 'ESC-50-master',
        'description': 'ESC-50 - 50 категорий окружающих звуков: машины, сирены, часы, клавиатура и др.',
        'compression': 'zip',
        'category': 'environmental'
    },
    'urbansound8k': {
        'url': 'https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1',
        'archive': 'UrbanSound8K.tar.gz',
        'extract_dir': 'UrbanSound8K',
        'description': 'UrbanSound8K - городские звуки: сирены, дрели, гудки, кондиционеры',
        'compression': 'gz',
        'category': 'environmental'
    },
}

MANUAL_DATASETS = {
    'm4singer': {
        'url': 'https://github.com/M4Singer/M4Singer',
        'description': 'M4Singer - Requires form submission at GitHub'
    },
    'clotho': {
        'url': 'https://zenodo.org/record/3490684',
        'description': 'Clotho - Environmental sounds (7z format, manual extraction needed)'
    }
}


def download_file(url, dest_path, chunk_size=8192):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    if dest_path.exists():
        print(f"  ✓ Already downloaded: {dest_path.name}")
        return dest_path
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  Downloading {dest_path.name}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="  Progress") as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
    
    return dest_path


def extract_archive(archive_path, extract_to, compression):
    """Extract tar or zip archive."""
    print(f"  Extracting {archive_path.name}...")
    
    if compression == 'zip':
        import zipfile
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    else:
        mode = f'r:{compression}'
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(extract_to)
    
    print(f"  ✓ Extracted to {extract_to}")


def download_dataset(name):
    """Download and extract a dataset."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False
    
    info = DATASETS[name]
    extract_dir = DATA_ROOT / info['extract_dir']
    
    print(f"\n{'='*50}")
    print(f"Dataset: {name}")
    print(f"Description: {info['description']}")
    print(f"{'='*50}")
    
    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"  ✓ Already exists: {extract_dir}")
        return True
    
    archive_path = DATA_ROOT / info['archive']
    
    try:
        download_file(info['url'], archive_path)
        extract_archive(archive_path, DATA_ROOT, info['compression'])
        print(f"  ✓ Done!")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def list_datasets():
    """List all available datasets."""
    print("\n=== Downloadable Datasets ===\n")
    for name, info in DATASETS.items():
        status = "✓" if (DATA_ROOT / info['extract_dir']).exists() else "○"
        print(f"  [{status}] {name}")
        print(f"      {info['description']}")
        print()
    
    print("\n=== Manual Download Required ===\n")
    for name, info in MANUAL_DATASETS.items():
        print(f"  [!] {name}")
        print(f"      {info['description']}")
        print(f"      URL: {info['url']}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Download audio datasets')
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--dataset', '-d', type=str, help='Download specific dataset')
    parser.add_argument('--list', '-l', action='store_true', help='List available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
    elif args.all:
        print("Downloading all datasets...")
        for name in DATASETS:
            download_dataset(name)
        print("\n" + "="*50)
        print("Note: M4Singer and Clotho require manual download.")
        print("="*50)
    elif args.dataset:
        download_dataset(args.dataset.lower())
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python download_datasets.py --list")
        print("  python download_datasets.py --dataset ljspeech")
        print("  python download_datasets.py --all")


if __name__ == '__main__':
    main()
