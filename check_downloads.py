import pandas as pd
from pathlib import Path
from pygbif import occurrences as occ

DATA_DIR = Path('occurrence-data')

def check_species(species_name, year_from=2015, countries=['US', 'CA', 'MX']):
    """Check if downloaded data matches expected count"""
    
    # Get expected count from GBIF
    try:
        expected = occ.search(
            scientificName=species_name,
            country=countries,
            hasCoordinate=True,
            year=f'{year_from},2025',
            limit=1
        ).get('count', 0)
    except:
        expected = 0
    
    # Get actual count from file
    safe_name = species_name.replace(' ', '-').lower()
    csv_file = DATA_DIR / safe_name / f'{safe_name}_gbif.csv'
    
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        actual = len(df)
        percent = (actual / expected * 100) if expected > 0 else 0
        
        # Determine status
        if percent >= 95:
            status = "✓"
        elif percent >= 80:
            status = "⚠"
        else:
            status = "✗"
        
        print(f"{status} {species_name:30s} | {actual:7,} / {expected:7,} ({percent:5.1f}%)")
        return {'species': species_name, 'actual': actual, 'expected': expected, 'percent': percent}
    else:
        print(f"✗ {species_name:30s} | FILE NOT FOUND")
        return {'species': species_name, 'actual': 0, 'expected': expected, 'percent': 0}

def main():
    # Read species list
    species_list_file = DATA_DIR / 'species-list.txt'
    
    if not species_list_file.exists():
        print("Error: species-list.txt not found")
        return
    
    species_list = [line.strip() for line in open(species_list_file) if line.strip()]
    
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETENESS CHECK")
    print("="*70)
    print(f"{'Status'} {'Species':30s} | {'Actual':>7s} / {'Expected':>7s} (Complete)")
    print("-"*70)
    
    results = []
    for species in species_list:
        result = check_species(species)
        results.append(result)
    
    print("="*70)
    
    # Summary
    complete = sum(1 for r in results if r['percent'] >= 95)
    partial = sum(1 for r in results if 80 <= r['percent'] < 95)
    incomplete = sum(1 for r in results if r['percent'] < 80)
    
    total_actual = sum(r['actual'] for r in results)
    total_expected = sum(r['expected'] for r in results)
    overall_percent = (total_actual / total_expected * 100) if total_expected > 0 else 0
    
    print(f"\nSummary:")
    print(f"  ✓ Complete (≥95%):   {complete}")
    print(f"  ⚠ Partial (80-95%):  {partial}")
    print(f"  ✗ Incomplete (<80%): {incomplete}")
    print(f"\nTotal: {total_actual:,} / {total_expected:,} records ({overall_percent:.1f}%)")
    print("="*70 + "\n")
    
    # List species to re-download
    need_redownload = [r['species'] for r in results if r['percent'] < 95]
    if need_redownload:
        print("Species needing re-download:")
        for sp in need_redownload:
            print(f"  • {sp}")

if __name__ == '__main__':
    main()