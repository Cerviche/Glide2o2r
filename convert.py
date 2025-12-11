#!/usr/bin/env python3
"""
Texture Pack Converter for SoH Texture Pipeline
===============================================
Converts texture packs using pre-generated maps.

Purpose:
--------
This script is the third step in a 3-step texture conversion pipeline:
1. hash.py: Calculate perceptual hashes for all textures (ONE-TIME, SLOW)
2. map.py: Create mapping between formats using cached hashes (FAST, REPEATABLE)
3. convert.py: Convert texture packs using generated maps

This script uses CSV maps created by map.py to convert texture packs from
Glide format to SoH format. It handles duplicate textures by copying to
all matching locations and provides comprehensive reporting.

Key Features:
-------------
- Uses pre-computed maps for fast conversion
- Copies textures to ALL matching locations (duplicate handling)
- Never moves or deletes original texture pack files
- Generates detailed CSV and JSON reports
- Provides conversion statistics and quality metrics

Usage:
------
python convert.py

Dependencies:
------------
- tqdm: For progress bars
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Third-party import for progress bars
from tqdm import tqdm


def find_texture_maps():
    """
    Find texture map CSV files created by map.py.

    Searches for files matching pattern 'texture_map_*.csv'.

    Returns:
        list or None: List of Path objects, or None if no maps found
    """
    map_files = list(Path('.').glob('texture_map_*.csv'))

    if not map_files:
        print("No texture map files found.")
        print("Please run map.py first to create a texture map.")
        return None

    # Sort by modification time, newest first
    return sorted(map_files, key=lambda x: x.stat().st_mtime, reverse=True)


def load_texture_map(map_file):
    """
    Load texture map from CSV file.

    Args:
        map_file (Path): Path to CSV map file

    Returns:
        tuple: (mapping_dict, stats_dict, metadata_dict)
            mapping_dict: {glide_filename: match_info_dict}
            stats_dict: Statistics about the loaded map
            metadata_dict: Metadata from map creation
    """
    print(f"Loading texture map: {map_file.name}")

    mapping = {}
    stats = {
        'total_mappings': 0,
        'hamming_distribution': Counter(),
        'duplicate_distribution': Counter(),
        'total_duplicate_paths': 0
    }

    # Read CSV file
    with open(map_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            glide_filename = row['glide_filename']

            # Convert string values to appropriate types
            match_info = {
                'soh_primary_path': row['soh_primary_path'],
                'soh_all_paths': row['soh_all_paths'],
                'hamming_distance': int(row['hamming_distance']),
                'glide_hash': row['glide_hash']
            }

            # Parse duplicate count if available
            if 'duplicate_count' in row and row['duplicate_count']:
                try:
                    match_info['duplicate_count'] = int(row['duplicate_count'])
                except (ValueError, TypeError):
                    match_info['duplicate_count'] = 1
            else:
                # Default to 1 if field not present
                match_info['duplicate_count'] = 1

            mapping[glide_filename] = match_info

            # Update statistics
            stats['total_mappings'] += 1
            stats['hamming_distribution'][match_info['hamming_distance']] += 1
            stats['duplicate_distribution'][match_info['duplicate_count']] += 1

            # Count total paths (including duplicates)
            if match_info['duplicate_count'] > 1:
                stats['total_duplicate_paths'] += (match_info['duplicate_count'] - 1)

    # Try to find corresponding JSON metadata file
    json_file = Path(str(map_file).replace('.csv', '.json'))
    metadata = {}
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f).get('metadata', {})
        except:
            metadata = {}

    # Print loading summary
    print(f"  ✓ Loaded {stats['total_mappings']} texture mappings")

    if stats['hamming_distribution']:
        print("  Match quality (Hamming distance):")
        for distance in sorted(stats['hamming_distribution'].keys()):
            count = stats['hamming_distribution'][distance]
            percentage = (count / stats['total_mappings']) * 100
            print(f"    Distance {distance}: {count} ({percentage:.1f}%)")

    if stats['duplicate_distribution']:
        unique_matches = stats['duplicate_distribution'].get(1, 0)
        duplicate_matches = stats['total_mappings'] - unique_matches

        print(f"  Duplicate handling:")
        print(f"    Unique matches (1 destination): {unique_matches}")
        print(f"    Duplicate matches (>1 destination): {duplicate_matches}")
        print(f"    Additional copies needed: {stats['total_duplicate_paths']}")

        if duplicate_matches > 0:
            print(f"    Average destinations per duplicate match: "
                  f"{stats['total_duplicate_paths'] / duplicate_matches + 1:.2f}")

    return mapping, stats, metadata


def scan_texture_pack(directory):
    """
    Recursively scan texture pack directory for PNG files.

    Args:
        directory (str): Path to texture pack directory

    Returns:
        list: List of tuples (relative_path, full_path, filename)
    """
    texture_files = []

    # Walk through directory tree
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Only process PNG files
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                # Get path relative to texture pack root
                rel_path = os.path.relpath(full_path, directory)
                texture_files.append((rel_path, full_path, file))

    return texture_files


def convert_texture_pack():
    """
    Main conversion function.

    This function:
    1. Lets user select a texture map
    2. Scans texture pack directory
    3. Copies matched textures to all SoH locations
    4. Generates comprehensive reports
    5. Never modifies original texture pack files

    Returns:
        None
    """
    print("=" * 60)
    print("TEXTURE PACK CONVERTER")
    print("Step 3 of 3 in texture conversion pipeline")
    print("=" * 60)
    print("This step converts texture packs using pre-generated maps")
    print("Original texture pack files are never modified or moved")
    print("=" * 60)

    # STEP 1: Find and select texture map
    print("\n--- Step 1: Select Texture Map ---")
    map_files = find_texture_maps()
    if not map_files:
        return  # No maps found

    # Display available maps
    print(f"\nAvailable texture maps (newest first):")
    display_count = min(5, len(map_files))

    for i, map_file in enumerate(map_files[:display_count], 1):
        size_kb = map_file.stat().st_size / 1024
        print(f"  {i}. {map_file.name} ({size_kb:.0f} KB)")

    if len(map_files) > display_count:
        print(f"  ... and {len(map_files) - display_count} more")

    # Get user selection
    print("\nOptions:")
    print("  # - Select by number (1-5)")
    print("  filename - Enter full filename")
    choice = input(f"\nSelect map (1-{display_count}) or enter filename: ").strip()

    if choice.isdigit() and 1 <= int(choice) <= len(map_files):
        selected_map = map_files[int(choice) - 1]
    else:
        selected_map = Path(choice)
        if not selected_map.exists():
            print(f"Error: Map file not found: {choice}")
            return

    # STEP 2: Load the selected map
    print("\n--- Step 2: Load Texture Map ---")
    mapping, map_stats, metadata = load_texture_map(selected_map)

    # Show map metadata if available
    if metadata:
        max_hamming = metadata.get('max_hamming', 'unknown')
        print(f"  Map created with Hamming distance ≤ {max_hamming}")

    # STEP 3: Get texture pack directory
    print("\n--- Step 3: Select Texture Pack ---")
    texture_pack_dir = input("Enter texture pack directory to convert: ").strip()

    # Validate directory
    if not os.path.exists(texture_pack_dir):
        print(f"Error: Texture pack directory not found: {texture_pack_dir}")
        return

    # STEP 4: Set up output directories
    print("\n--- Step 4: Output Configuration ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        'output_dir': f"converted_{timestamp}",
        'csv_report': f"conversion_report_{timestamp}.csv",
        'json_debug': f"conversion_report_{timestamp}.json"
    }

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Output directory: {config['output_dir']}")
    print(f"CSV report: {config['csv_report']}")
    print(f"JSON debug: {config['json_debug']}")

    if map_stats['total_duplicate_paths'] > 0:
        print(f"\nNote: This map includes duplicate destinations")
        print(f"      Textures will be copied to ALL matching locations")

    # STEP 5: Scan texture pack
    print("\n--- Step 5: Scan Texture Pack ---")
    texture_pack_files = scan_texture_pack(texture_pack_dir)
    print(f"Found {len(texture_pack_files)} PNG files in texture pack")

    # STEP 6: Initialize conversion tracking
    conversion_stats = {
        'total_textures': len(texture_pack_files),
        'converted': 0,
        'missing': 0,
        'errors': 0,
        'total_copies_made': 0,
        'duplicate_copies': 0,
        'hamming_distribution': Counter(),
        'duplicate_distribution': Counter(),
        'converted_details': [],
        'missing_details': [],
        'error_details': []
    }

    # STEP 7: Convert textures
    print("\n--- Step 6: Convert Textures ---")
    print(f"Converting textures to: {config['output_dir']}")

    for rel_path, full_path, filename in tqdm(texture_pack_files, desc="Converting", unit="files"):
        try:
            if filename in mapping:
                match_info = mapping[filename]
                hamming = match_info['hamming_distance']

                # Get all destination paths
                if match_info['soh_all_paths']:
                    all_soh_paths = match_info['soh_all_paths'].split('|')
                else:
                    all_soh_paths = [match_info['soh_primary_path']]

                duplicate_count = len(all_soh_paths)

                # Create all destination directories
                copied_paths = []
                for soh_path in all_soh_paths:
                    dest_path = os.path.join(config['output_dir'], soh_path)
                    dest_dir = os.path.dirname(dest_path)
                    os.makedirs(dest_dir, exist_ok=True)

                    # Copy file (never move original)
                    shutil.copy2(full_path, dest_path)
                    copied_paths.append(dest_path)

                # Update statistics
                conversion_stats['converted'] += 1
                conversion_stats['total_copies_made'] += duplicate_count
                conversion_stats['duplicate_copies'] += (duplicate_count - 1)
                conversion_stats['hamming_distribution'][hamming] += 1
                conversion_stats['duplicate_distribution'][duplicate_count] += 1

                # Record conversion details
                conversion_details = {
                    'original_file': rel_path,
                    'filename': filename,
                    'soh_primary_path': all_soh_paths[0],
                    'soh_all_paths': '|'.join(all_soh_paths),
                    'duplicate_count': duplicate_count,
                    'additional_copies': duplicate_count - 1,
                    'hamming_distance': hamming,
                    'source': full_path,
                    'all_destinations': '|'.join(copied_paths),
                    'status': 'CONVERTED',
                    'error': '',
                    'reason': ''
                }
                conversion_stats['converted_details'].append(conversion_details)

            else:
                # No mapping found
                conversion_stats['missing'] += 1

                missing_details = {
                    'original_file': rel_path,
                    'filename': filename,
                    'soh_primary_path': '',
                    'soh_all_paths': '',
                    'duplicate_count': '',
                    'additional_copies': '',
                    'hamming_distance': '',
                    'source': full_path,
                    'all_destinations': '',
                    'status': 'MISSING',
                    'error': '',
                    'reason': 'No match found in texture map'
                }
                conversion_stats['missing_details'].append(missing_details)

        except Exception as e:
            # Handle conversion errors
            conversion_stats['errors'] += 1

            error_details = {
                'original_file': rel_path,
                'filename': filename,
                'soh_primary_path': '',
                'soh_all_paths': '',
                'duplicate_count': '',
                'additional_copies': '',
                'hamming_distance': '',
                'source': full_path,
                'all_destinations': '',
                'status': 'ERROR',
                'error': str(e),
                'reason': str(e)
            }
            conversion_stats['error_details'].append(error_details)
            print(f"  Error converting {rel_path}: {e}")

    # STEP 8: Generate reports
    print("\n--- Step 7: Generate Reports ---")

    # Define all CSV fields
    csv_fields = [
        'original_file',
        'filename',
        'soh_primary_path',
        'soh_all_paths',
        'duplicate_count',
        'additional_copies',
        'hamming_distance',
        'source',
        'all_destinations',
        'status',
        'error',
        'reason'
    ]

    # Save CSV report
    with open(config['csv_report'], 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()

        # Write all records
        all_records = (conversion_stats['converted_details'] +
                      conversion_stats['missing_details'] +
                      conversion_stats['error_details'])

        for record in all_records:
            # Ensure all fields exist
            for field in csv_fields:
                if field not in record:
                    record[field] = ''
            writer.writerow(record)

    # Save JSON debug info
    debug_info = {
        'timestamp': datetime.now().isoformat(),
        'map_used': selected_map.name,
        'texture_pack': texture_pack_dir,
        'output_directory': config['output_dir'],
        'map_statistics': {
            'total_mappings': map_stats['total_mappings'],
            'hamming_distribution': dict(map_stats['hamming_distribution']),
            'duplicate_distribution': dict(map_stats['duplicate_distribution']),
            'total_duplicate_paths': map_stats['total_duplicate_paths']
        },
        'conversion_statistics': {
            'total_textures': conversion_stats['total_textures'],
            'converted': conversion_stats['converted'],
            'missing': conversion_stats['missing'],
            'errors': conversion_stats['errors'],
            'total_copies_made': conversion_stats['total_copies_made'],
            'duplicate_copies': conversion_stats['duplicate_copies'],
            'hamming_distribution': dict(conversion_stats['hamming_distribution']),
            'duplicate_distribution': dict(conversion_stats['duplicate_distribution']),
            'conversion_rate': f"{(conversion_stats['converted'] / conversion_stats['total_textures']) * 100:.1f}%"
                              if conversion_stats['total_textures'] > 0 else "0%"
        }
    }

    with open(config['json_debug'], 'w', encoding='utf-8') as jsonfile:
        json.dump(debug_info, jsonfile, indent=2, default=str)

    # STEP 9: Print summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Texture pack files: {conversion_stats['total_textures']}")
    print(f"Successfully converted: {conversion_stats['converted']}")
    print(f"No mapping found: {conversion_stats['missing']}")
    print(f"Errors during conversion: {conversion_stats['errors']}")

    if conversion_stats['converted'] > 0:
        conversion_rate = (conversion_stats['converted'] / conversion_stats['total_textures']) * 100
        print(f"Conversion rate: {conversion_rate:.1f}%")

        print(f"\nMatch quality (Hamming distance):")
        for distance in sorted(conversion_stats['hamming_distribution'].keys()):
            count = conversion_stats['hamming_distribution'][distance]
            percentage = (count / conversion_stats['converted']) * 100
            print(f"  Distance {distance}: {count} ({percentage:.1f}%)")

        if conversion_stats['duplicate_copies'] > 0:
            print(f"\nDuplicate handling:")
            print(f"  Additional copies made: {conversion_stats['duplicate_copies']}")
            print(f"  Total file operations: {conversion_stats['total_copies_made']}")
            print(f"  Average destinations per texture: "
                  f"{conversion_stats['total_copies_made'] / conversion_stats['converted']:.2f}")

    print(f"\nOutput files:")
    print(f"  Converted textures: {config['output_dir']}")
    print(f"  CSV report: {config['csv_report']}")
    print(f"  JSON debug: {config['json_debug']}")

    print(f"\nIMPORTANT:")
    print(f"  Original texture pack files are UNMODIFIED")
    print(f"  Converted textures are in: {config['output_dir']}")

    if conversion_stats['missing'] > 0:
        print(f"\n⚠️  {conversion_stats['missing']} textures had no mapping")
        print("  These may be unique to this texture pack")
        print("  Check CSV report for details")


if __name__ == "__main__":
    """
    Main entry point with error handling.
    """
    try:
        convert_texture_pack()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
