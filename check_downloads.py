import pandas as pd
import geopandas as gpd
from pathlib import Path
from pygbif import occurrences as occ

DATA_DIR = Path('occurrence-data')
METADATA_FILE = DATA_DIR / 'species-metadata.csv'

def check_species(row):
    """Check if downloaded data matches expected count from metadata"""
    
    species_name = row['species_name']
    expected = row.get('expected_obs', 0)
    expected = 0 if pd.isna(expected) else int(expected)
    
    # Get actual count from GeoJSON file
    safe_name = species_name.replace(' ', '-').lower()
    geojson_file = DATA_DIR / f'{safe_name}-gbif.geojson'
    
    if geojson_file.exists():
        try:
            gdf = gpd.read_file(geojson_file)
            actual = len(gdf)
            percent = (actual / expected * 100) if expected > 0 else 100.0  # If expected is 0, assume 100%
            
            # Determine status
            if expected == 0 or percent >= 95:
                status = "✓"
            elif percent >= 80:
                status = "⚠"
            else:
                status = "✗"
            
            expected_str = f"{expected:,}" if expected > 0 else "N/A"
            print(f"{status} {species_name:30s} | {actual:7,} / {expected_str:>7s} ({percent:5.1f}%)")
            return {'species': species_name, 'actual': actual, 'expected': expected, 'percent': percent, 'exists': True}
        except Exception as e:
            print(f"✗ {species_name:30s} | ERROR READING FILE: {e}")
            return {'species': species_name, 'actual': 0, 'expected': expected, 'percent': 0, 'exists': False}
    else:
        expected_str = f"{expected:,}" if expected > 0 else "N/A"
        print(f"✗ {species_name:30s} | FILE NOT FOUND (expected {expected_str})")
        return {'species': species_name, 'actual': 0, 'expected': expected, 'percent': 0, 'exists': False}

def main():
    # Read metadata CSV
    if not METADATA_FILE.exists():
        print("Error: species-metadata.csv not found in occurrence-data/")
        print("Run 'python procure.py' first to create it")
        return
    
    df = pd.read_csv(METADATA_FILE)
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETENESS CHECK")
    print("="*70)
    print(f"{'Status'} {'Species':30s} | {'Actual':>7s} / {'Expected':>7s} (Complete)")
    print("-"*70)
    
    results = []
    for _, row in df.iterrows():
        result = check_species(row)
        results.append(result)
    
    print("="*70)
    
    # Summary
    complete = sum(1 for r in results if r['exists'] and (r['expected'] == 0 or r['percent'] >= 95))
    partial = sum(1 for r in results if r['exists'] and r['expected'] > 0 and 80 <= r['percent'] < 95)
    incomplete = sum(1 for r in results if r['exists'] and r['expected'] > 0 and r['percent'] < 80)
    missing = sum(1 for r in results if not r['exists'])
    
    total_actual = sum(r['actual'] for r in results)
    total_expected = sum(r['expected'] for r in results if r['expected'] > 0)
    overall_percent = (total_actual / total_expected * 100) if total_expected > 0 else 0
    
    print(f"\nSummary:")
    print(f"  ✓ Complete (≥95%):   {complete}")
    print(f"  ⚠ Partial (80-95%):  {partial}")
    print(f"  ✗ Incomplete (<80%): {incomplete}")
    print(f"  ✗ Missing files:     {missing}")
    print(f"\nTotal: {total_actual:,} / {total_expected:,} records ({overall_percent:.1f}%)")
    print("="*70 + "\n")
    
    # List species to re-download (only those with missing files or truly incomplete)
    need_redownload = [r['species'] for r in results if not r['exists'] or (r['expected'] > 0 and r['percent'] < 95)]
    if need_redownload:
        print("Species needing download/re-download:")
        for sp in need_redownload:
            print(f"  • {sp}")
        print()
    
    # Show metadata status comparison
    print("="*70)
    print("METADATA CSV STATUS COMPARISON")
    print("="*70)
    
    for status in ['complete', 'pending', 'error']:
        subset = df[df['status'] == status]
        if len(subset) > 0:
            print(f"\n{status.upper()} ({len(subset)}):")
            for _, row in subset.iterrows():
                if status == 'complete':
                    quality = row['data_quality'] if pd.notna(row['data_quality']) else 0
                    print(f"  {row['species_name']}: {row['actual_obs']:,} obs ({quality:.1f}%)")
                else:
                    print(f"  {row['species_name']}")
    
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    main()