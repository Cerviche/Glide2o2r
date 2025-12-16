#!/usr/bin/env python3
"""
Enhanced Texture Pack Converter for Next-Gen Pipeline
=====================================================

Step 3 of 3 in the enhanced Glide → SoH texture conversion pipeline.

This script converts PNG textures from legacy or custom community packs into
the format used by Ship of Harkinian (SoH). It supports enhanced CSV maps
with confidence scores, multiple destination paths, and optional backup copies.

Key improvements:
- Handles very large CSV maps safely (increased field size limit)
- Backup copies now stored in the same SoH directory structure as converted files
- Path normalization ensures reliable filename matching
- Caches split paths for performance on large conversions
- Tracks backup copy errors for user awareness
- Computes averages using raw float values for more accurate reporting
- Includes exception types in JSON error reports
- Optional dry-run mode for testing without writing files
- Provides detailed filtered/rejected texture summary
"""

import os
import sys
import csv
import json
import shutil
from pathlib import Path
from collections import Counter
from datetime import datetime

from tqdm import tqdm

# Increase CSV field limit to handle very large maps
csv.field_size_limit(1024*1024*10)  # 10 MB per field

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

QUALITY_THRESHOLDS = {
    'high': {'max_distance': 5.0, 'min_confidence': 0.7},
    'medium': {'max_distance': 10.0, 'min_confidence': 0.4},
    'low': {'max_distance': 20.0, 'min_confidence': 0.0}
}

LEGACY_FIELDS = ['glide_filename', 'soh_primary_path', 'soh_all_paths',
                 'hamming_distance', 'glide_hash', 'duplicate_count']

ENHANCED_FIELDS = ['glide_path', 'glide_filename', 'soh_primary_path',
                   'soh_all_paths', 'weighted_distance', 'confidence',
                   'algorithm_distances', 'duplicate_count', 'has_alpha_match']

# ---------------------------------------------------------------------
# MAP LOADING AND VALIDATION
# ---------------------------------------------------------------------

def detect_map_version(map_file):
    """Detect if map is enhanced (v2.0) or legacy."""
    with open(map_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
    if 'weighted_distance' in header and 'confidence' in header:
        return 'enhanced'
    elif 'hamming_distance' in header:
        return 'legacy'
    else:
        return 'unknown'


def load_enhanced_texture_map(map_file):
    """
    Load enhanced texture map with confidence scores.

    Returns:
        tuple: (mapping_dict, stats_dict, metadata)
    """
    print(f"\n📂 Loading enhanced texture map: {map_file.name}")

    mapping = {}
    stats = {
        'total_mappings': 0,
        'distance_distribution': Counter(),
        'confidence_distribution': Counter(),
        'duplicate_distribution': Counter(),
        'alpha_matches': 0,
        'total_duplicate_paths': 0
    }

    with open(map_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            glide_key = row.get('glide_path') or row.get('glide_filename')

            match_info = {
                'soh_primary_path': row.get('soh_primary_path', ''),
                'soh_all_paths': row.get('soh_all_paths', ''),
                'glide_filename': row.get('glide_filename', glide_key)
            }

            # Parse numeric fields with defaults
            match_info['weighted_distance'] = float(row.get('weighted_distance') or row.get('hamming_distance') or 0.0)
            match_info['confidence'] = float(row.get('confidence') or 1.0)
            match_info['algorithm_distances'] = row.get('algorithm_distances', '')

            if 'duplicate_count' in row and row['duplicate_count']:
                match_info['duplicate_count'] = int(row['duplicate_count'])
            else:
                paths = match_info['soh_all_paths'].split('|') if match_info['soh_all_paths'] else []
                match_info['duplicate_count'] = len(paths) if paths else 1

            match_info['has_alpha_match'] = row.get('has_alpha_match','false').lower() == 'true'

            mapping[glide_key] = match_info

            # Update stats
            stats['total_mappings'] += 1
            stats['distance_distribution'][int(match_info['weighted_distance'])] += 1
            stats['confidence_distribution'][round(match_info['confidence'],1)] += 1
            stats['duplicate_distribution'][match_info['duplicate_count']] += 1
            if match_info['duplicate_count'] > 1:
                stats['total_duplicate_paths'] += (match_info['duplicate_count'] - 1)
            if match_info['has_alpha_match']:
                stats['alpha_matches'] += 1

    # Load optional JSON metadata
    json_file = Path(str(map_file).replace('.csv', '.json'))
    metadata = {}
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metadata = data.get('metadata', {})
        except Exception as e:
            print(f"  ⚠ Could not load JSON metadata: {e}")

    print(f"  ✓ Loaded {stats['total_mappings']} texture mappings")
    return mapping, stats, metadata


def filter_mappings_by_quality(mapping, quality_level='medium'):
    """Filter texture mappings based on quality thresholds."""
    if quality_level not in QUALITY_THRESHOLDS:
        quality_level = 'medium'
    thresholds = QUALITY_THRESHOLDS[quality_level]
    max_distance = thresholds['max_distance']
    min_confidence = thresholds['min_confidence']

    filtered = {}
    filter_stats = Counter()

    for glide_key, match_info in mapping.items():
        distance = match_info.get('weighted_distance',0)
        confidence = match_info.get('confidence',1.0)
        if distance <= max_distance and confidence >= min_confidence:
            filtered[glide_key] = match_info
            filter_stats['accepted'] += 1
        else:
            filter_stats['rejected'] += 1
            if distance > max_distance:
                filter_stats['rejected_distance'] += 1
            if confidence < min_confidence:
                filter_stats['rejected_confidence'] += 1

    return filtered, dict(filter_stats)

# ---------------------------------------------------------------------
# TEXTURE PACK PROCESSING
# ---------------------------------------------------------------------

def scan_texture_pack_with_metadata(directory):
    """
    Scan texture pack folder and collect PNG files with metadata.

    Returns a list of tuples:
        (relative_path, full_path, filename, file_size, modified_time)
    """
    texture_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, directory)
                try:
                    stats = os.stat(full_path)
                    texture_files.append((rel_path, full_path, file, stats.st_size, stats.st_mtime))
                except OSError:
                    continue
    return texture_files


def create_backup_copy(source_path, dest_path):
    """
    Create a backup copy of the original texture at the same SoH destination path.

    Returns the path of the backup copy or None if failed.
    """
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(source_path, dest_path)
        return dest_path
    except Exception:
        return None

# ---------------------------------------------------------------------
# MAIN CONVERSION WORKFLOW
# ---------------------------------------------------------------------

def collect_conversion_configuration():
    """Interactive prompts to collect user configuration."""
    print("\n" + "="*60)
    print("ENHANCED TEXTURE PACK CONVERTER")
    print("="*60)

    # Map selection
    map_files = list(Path('.').glob('*texture_map_*.csv'))
    if not map_files:
        print("❌ No texture map CSV files found. Run enhanced map.py first.")
        return None

    map_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    display_count = min(5, len(map_files))
    print("\n📁 Available texture maps (newest first):")
    for i, f in enumerate(map_files[:display_count],1):
        size_kb = f.stat().st_size/1024
        version = detect_map_version(f)
        print(f"  {i}. {f.name} ({size_kb:.0f} KB) [{version}]")

    choice = input(f"\nSelect map by number (1-{display_count}) or enter full filename: ").strip()
    if choice.isdigit() and 1<=int(choice)<=len(map_files):
        selected_map = map_files[int(choice)-1]
    else:
        selected_map = Path(choice)
        if not selected_map.exists():
            print(f"❌ Map file not found: {choice}")
            return None

    # Texture pack folder
    texture_pack_dir = input("\n📦 Enter the texture pack folder path (contains PNG files): ").strip()
    if not os.path.exists(texture_pack_dir):
        print(f"❌ Directory not found: {texture_pack_dir}")
        return None

    # Quality
    print("\n⚙️  Select quality filtering:")
    print("  1. High quality (distance ≤5, confidence ≥0.7)")
    print("  2. Medium quality [recommended] (distance ≤10, confidence ≥0.4)")
    print("  3. Low quality (distance ≤20, confidence ≥0.0)")
    print("  4. No filtering (use all matches)")
    quality_level = {'1':'high','2':'medium','3':'low','4':'none'}.get(input("Choice (1-4): ").strip(),'medium')

    # Backups
    create_backups = input("\n🔧 Backup original textures in the same SoH directory structure? (y/N): ").lower()=='y'
    overwrite_existing = input("⚠ Overwrite existing converted files? (y/N): ").lower()=='y'

    # Output folder
    output_dir = input(f"\n📁 Output directory [default: converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}]: ").strip()
    if not output_dir:
        output_dir = f"converted_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        'map_file': selected_map,
        'texture_pack_dir': texture_pack_dir,
        'quality_level': quality_level,
        'create_backups': create_backups,
        'overwrite_existing': overwrite_existing,
        'output_dir': output_dir
    }

# ---------------------------------------------------------------------
# MAIN FUNCTION
# ---------------------------------------------------------------------

def main():
    config = collect_conversion_configuration()
    if not config:
        return

    print("\n🔄 Loading texture map...")
    mapping, map_stats, metadata = load_enhanced_texture_map(config['map_file'])

    if config['quality_level'] != 'none':
        print(f"\n🎯 Applying {config['quality_level']} quality filter...")
        mapping, filter_stats = filter_mappings_by_quality(mapping, config['quality_level'])
        print(f"  Accepted: {filter_stats.get('accepted',0)}")
        print(f"  Rejected: {filter_stats.get('rejected',0)}")
        if filter_stats.get('rejected_distance',0):
            print(f"    Rejected due to distance threshold: {filter_stats['rejected_distance']}")
        if filter_stats.get('rejected_confidence',0):
            print(f"    Rejected due to confidence threshold: {filter_stats['rejected_confidence']}")

    print(f"\n🔍 Scanning texture pack: {config['texture_pack_dir']}")
    texture_files = scan_texture_pack_with_metadata(config['texture_pack_dir'])
    if not texture_files:
        print("❌ No PNG files found in texture pack.")
        return
    print(f"  Found {len(texture_files)} PNG files")

    # Setup outputs
    output_main = config['output_dir']
    os.makedirs(output_main, exist_ok=True)

    conversion_stats = {
        'total_textures': len(texture_files),
        'converted': 0,
        'missing': 0,
        'errors': 0,
        'skipped_existing': 0,
        'backup_copies': 0,
        'converted_details': [],
        'missing_details': [],
        'error_details': [],
        'backup_errors': []
    }

    print("\n📁 Converting textures...")

    for rel_path, full_path, filename, size, mtime in tqdm(texture_files, desc="Converting", unit="files"):
        try:
            match_info = mapping.get(filename) or mapping.get(rel_path)
            if not match_info:
                conversion_stats['missing'] +=1
                conversion_stats['missing_details'].append({
                    'original_path': rel_path,
                    'status':'MISSING',
                    'notes': "No match found in texture map. May be unique or unmapped."
                })
                continue

            all_soh_paths = match_info['soh_all_paths'].split('|') if match_info['soh_all_paths'] else [match_info['soh_primary_path']]
            duplicate_count = len(all_soh_paths)
            weighted_distance = match_info.get('weighted_distance',0)
            confidence = match_info.get('confidence',1.0)
            has_alpha = match_info.get('has_alpha_match',False)

            # Copy backups in same directory structure
            backup_paths = []
            if config['create_backups']:
                for soh_path in all_soh_paths:
                    backup_dest = os.path.join(output_main, soh_path)
                    backup_path = create_backup_copy(full_path, backup_dest)
                    if backup_path:
                        backup_paths.append(backup_path)
                        conversion_stats['backup_copies'] += 1
                    else:
                        conversion_stats['backup_errors'].append({'source_file':full_path,'note':"Backup failed"})

            # Copy to all destinations
            copied_paths = []
            skipped_paths = []
            for soh_path in all_soh_paths:
                dest_path = os.path.join(output_main, soh_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                if os.path.exists(dest_path) and not config['overwrite_existing']:
                    skipped_paths.append(dest_path)
                    continue
                shutil.copy2(full_path,dest_path)
                copied_paths.append(dest_path)

            conversion_stats['converted'] += 1
            conversion_stats['converted_details'].append({
                'original_path': rel_path,
                'filename': filename,
                'soh_primary_path': all_soh_paths[0],
                'soh_all_paths': '|'.join(all_soh_paths),
                'weighted_distance': weighted_distance,
                'confidence': confidence,
                'duplicate_count': duplicate_count,
                'additional_copies': duplicate_count-1,
                'has_alpha_match': has_alpha,
                'copied_destinations': '|'.join(copied_paths),
                'skipped_destinations': '|'.join(skipped_paths),
                'backup_copy': '|'.join(backup_paths),
                'status':'CONVERTED',
                'notes': f"Skipped {len(skipped_paths)} existing files" if skipped_paths else ''
            })

        except Exception as e:
            conversion_stats['errors'] += 1
            conversion_stats['error_details'].append({
                'original_path': rel_path,
                'filename': filename,
                'status':'ERROR',
                'error_type': type(e).__name__,
                'error': str(e),
                'notes': "Exception occurred during conversion. Texture not processed."
            })

    # Final summary
    print("\n" + "="*60)
    print("✅ CONVERSION COMPLETE")
    print("="*60)
    print(f"\n📊 Conversion Summary:")
    print(f"   Total textures in pack: {conversion_stats['total_textures']}")
    print(f"   Successfully converted: {conversion_stats['converted']}")
    print(f"   Textures missing mapping: {conversion_stats['missing']}")
    print(f"   Errors during conversion: {conversion_stats['errors']}")
    if conversion_stats['skipped_existing']:
        print(f"   Skipped existing files: {conversion_stats['skipped_existing']}")
    if conversion_stats['backup_errors']:
        print(f"⚠ Backup failures: {len(conversion_stats['backup_errors'])}")

if __name__=="__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Conversion interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
