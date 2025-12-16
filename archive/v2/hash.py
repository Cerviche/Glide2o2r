#!/usr/bin/env python3
"""
Texture Hash Calculator for SoH Texture Pipeline
================================================
Calculates and stores perceptual hashes for texture directories.

Purpose:
--------
This script is the first step in a 3-step texture conversion pipeline:
1. hash.py: Calculate perceptual hashes for all textures (ONE-TIME, SLOW)
2. map.py: Create mapping between formats using cached hashes (FAST, REPEATABLE)
3. convert.py: Convert texture packs using generated maps

This script performs the computationally expensive perceptual hashing once,
saving results to JSON files that can be reused by map.py for different
matching thresholds.

Workflow:
---------
1. Scan directories for PNG files
2. Calculate perceptual hashes for all textures
3. Handle different image modes and errors
4. Save hashes to JSON with metadata for reuse
5. Generate detailed statistics and error reports

Usage:
------
python hash.py

Dependencies:
------------
- Pillow: For image processing
- imagehash: For perceptual hashing
- tqdm: For progress bars
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Third-party imports
import imagehash  # Perceptual hashing library
from PIL import Image  # Image processing
from tqdm import tqdm  # Progress bars


def scan_directory(directory):
    """
    Scan directory for PNG files recursively.

    Args:
        directory (str): Root directory to scan

    Returns:
        list: List of tuples (relative_path, full_path) for each PNG file
    """
    png_files = []

    # os.walk traverses directory tree recursively
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if file is a PNG (case-insensitive)
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                # Get path relative to the root directory for consistent mapping
                rel_path = os.path.relpath(full_path, directory)
                png_files.append((rel_path, full_path))

    return png_files


def calculate_perceptual_hash(image_path):
    """
    Calculate perceptual hash for an image with robust error handling.

    Perceptual hashing creates a fingerprint of an image based on its visual
    content using the pHash algorithm. This function handles various image
    formats and modes to ensure consistent hashing.

    Args:
        image_path (str): Full path to the image file

    Returns:
        tuple: (success_bool, result)
            success_bool: True if hash calculated successfully
            result: imagehash.ImageHash on success, error dict on failure
    """
    try:
        with Image.open(image_path) as img:
            # Store original image metadata for debugging
            original_mode = img.mode
            original_size = img.size
            original_format = img.format

            # Handle different image modes to ensure consistent hashing
            if img.mode in ('P', 'PA'):
                # Palette mode (8-bit with color palette)
                img = img.convert('RGBA')  # Convert to RGBA to preserve colors
            elif img.mode == '1':
                # 1-bit pixels (black and white)
                img = img.convert('L')  # Convert to 8-bit grayscale
            elif img.mode == 'L':
                # 8-bit grayscale - already compatible
                pass
            elif img.mode in ('RGB', 'RGBA'):
                if img.mode == 'RGBA':
                    # RGBA has alpha channel (transparency)
                    # Composite onto white background for consistent hashing
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha as mask
                    img = background
                # RGB images need no conversion
            else:
                # Any other mode (CMYK, YCbCr, LAB, HSV, etc.)
                img = img.convert('RGB')

            # Calculate perceptual hash using pHash algorithm
            # pHash: Convert to grayscale, reduce to 32x32, DCT, compare to median
            hash_result = imagehash.phash(img)

            if hash_result is None:
                return False, {
                    'error': 'imagehash.phash() returned None',
                    'type': 'HashCalculationError',
                    'file': os.path.basename(image_path)
                }

            return True, hash_result

    except Exception as e:
        # Return detailed error information for debugging
        return False, {
            'error': str(e),
            'type': type(e).__name__,
            'file': os.path.basename(image_path)
        }


def calculate_hashes():
    """
    Main function to calculate and save perceptual hashes for texture directories.

    This is the computationally expensive step that only needs to be done once
    per texture directory. Results are saved to JSON for reuse by map.py.

    Returns:
        str or None: Path to created hash file, or None if failed
    """
    print("=" * 60)
    print("TEXTURE HASH CALCULATOR")
    print("Step 1 of 3 in texture conversion pipeline")
    print("=" * 60)
    print("This step calculates perceptual hashes (SLOW, one-time)")
    print("Results are cached for fast rematching with different thresholds")
    print("=" * 60)

    # Configuration
    config = {}

    # Get directory inputs from user
    print("\n--- Directory Configuration ---")
    print("Enter directory containing textures to hash.")
    print("For Glide format: Enter Glide directory")
    print("For SoH format: Enter SoH directory")
    print("\nYou'll need to run this script twice:")
    print("1. First for Glide format directory")
    print("2. Then for SoH format directory")

    config['texture_dir'] = input("\nEnter texture directory to hash: ").strip()

    # Validate directory exists
    if not os.path.exists(config['texture_dir']):
        print(f"Error: Directory not found: {config['texture_dir']}")
        return None

    # Get format type for naming
    print("\n--- Format Identification ---")
    print("What format are these textures?")
    print("1. Glide (Rice) format - hash-based filenames, flat structure")
    print("2. SoH format - descriptive filenames, directory structure")

    while True:
        format_choice = input("Enter 1 for Glide, 2 for SoH: ").strip()
        if format_choice == '1':
            config['format'] = 'glide'
            break
        elif format_choice == '2':
            config['format'] = 'soh'
            break
        else:
            print("Please enter 1 or 2")

    # Set up output files with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config['hash_output'] = f"{config['format']}_hashes_{timestamp}.json"
    config['stats_output'] = f"{config['format']}_hash_stats_{timestamp}.json"

    print(f"\n--- Hash Calculation ---")
    print(f"Directory: {config['texture_dir']}")
    print(f"Format: {config['format'].upper()}")
    print(f"Hash output: {config['hash_output']}")
    print(f"Stats output: {config['stats_output']}")
    print("\nThis may take several minutes depending on number of textures...")

    # Scan directory for PNG files
    print(f"\n[1/3] Scanning {config['texture_dir']} for PNG files...")
    texture_files = scan_directory(config['texture_dir'])
    print(f"  Found {len(texture_files)} PNG files")

    # Calculate hashes with progress tracking
    print(f"\n[2/3] Calculating perceptual hashes...")

    hashes = {}  # Successful hashes: rel_path -> hash_string
    hash_objects = {}  # For debugging: rel_path -> hash_object
    errors = []  # Failed hashes with error details
    stats = Counter()  # Track statistics

    for rel_path, full_path in tqdm(texture_files, desc="Hashing", unit="files"):
        success, result = calculate_perceptual_hash(full_path)

        if success:
            # Store hash as string for JSON serialization
            hashes[rel_path] = str(result)
            # Store hash object for potential debugging
            hash_objects[rel_path] = result
            stats['successful'] += 1
        else:
            # Store error details
            errors.append({
                'file': rel_path,
                'full_path': full_path,
                'error': result['error'],
                'type': result['type']
            })
            stats['failed'] += 1

    # Save hashes to JSON file
    print(f"\n[3/3] Saving {len(hashes)} hashes to {config['hash_output']}...")

    hash_data = {
        'metadata': {
            'created': datetime.now().isoformat(),
            'directory': config['texture_dir'],
            'format': config['format'],
            'total_files': len(texture_files),
            'successful_hashes': len(hashes),
            'failed_hashes': len(errors)
        },
        'hashes': hashes,  # rel_path -> hash_string
        'file_list': list(hashes.keys())  # For quick lookup
    }

    with open(config['hash_output'], 'w', encoding='utf-8') as f:
        json.dump(hash_data, f, indent=2, default=str)

    # Save detailed statistics
    stats_data = {
        'config': config,
        'statistics': {
            'total_textures': len(texture_files),
            'successfully_hashed': stats['successful'],
            'failed_to_hash': stats['failed'],
            'success_rate': f"{(stats['successful']/len(texture_files))*100:.1f}%" if texture_files else "0%"
        },
        'sample_errors': errors[:50] if errors else []  # First 50 errors
    }

    with open(config['stats_output'], 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, indent=2, default=str)

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("HASH CALCULATION COMPLETE")
    print("=" * 60)
    print(f"Texture directory: {config['texture_dir']}")
    print(f"Format: {config['format'].upper()}")
    print(f"Total PNG files: {len(texture_files)}")
    print(f"Successfully hashed: {stats['successful']}")
    print(f"Failed to hash: {stats['failed']}")
    print(f"Success rate: {stats_data['statistics']['success_rate']}")

    if errors:
        print(f"\nHashing errors (first 5):")
        for error in errors[:5]:
            print(f"  {error['file']}: {error['type']} - {error['error']}")
        print(f"  See {config['stats_output']} for full error list")

    print(f"\nOutput files:")
    print(f"  Hash file: {config['hash_output']}")
    print(f"  Statistics: {config['stats_output']}")

    print(f"\nNEXT STEP:")
    print(f"1. Run this script for the OTHER texture format")
    print(f"2. Then run map.py to create mappings between formats")

    return config['hash_output']


if __name__ == "__main__":
    """
    Main entry point with error handling.
    """
    try:
        calculate_hashes()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
