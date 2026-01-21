from pygbif import occurrences as occ
import pandas as pd
import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
from tqdm import tqdm

# Setup species sub-directory if it doesn't exist
def setup_species_folder(species_name):
    safe_name = species_name.replace(' ', '-').lower()
    species_dir = Path('occurrence-data') / safe_name
    species_dir.mkdir(parents=True, exist_ok=True)
    return species_dir, safe_name

# Saves download progress if interrupted 
def save_checkpoint(species_dir, offset, total_records):
    checkpoint = {
        'offset': offset,
        'total_records': total_records,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    checkpoint_file = species_dir / '.checkpoint.json'
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)

# Searches + initializes script from last checkpoint
def load_checkpoint(species_dir):
    checkpoint_file = species_dir / '.checkpoint.json'
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return None

# Determine total available records for species
def get_total_count(species_name, year_from=2015):
    try:
        result = occ.search(
            scientificName=species_name,
            country=['US', 'CA', 'MX'],
            hasCoordinate=True,
            year=f'{year_from},2025',
            limit=1
        )
        return result.get('count', 0)
    except Exception as e:
        print(f"[{species_name}] Error getting count: {e}")
        return None

# Occurrence data procurement function (single threaded)
def download_species(species_name, year_from=2015 ):
    # Setup folder structure
    species_dir, safe_name = setup_species_folder(species_name)
    output_file = species_dir / f'{safe_name}_gbif.csv'
    temp_file = species_dir / f'{safe_name}_temp.csv'
    
    all_records = []
    offset = 0
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(species_dir)
    if checkpoint:
        offset = checkpoint['offset']
        # Load existing temp data
        if temp_file.exists():
            existing = pd.read_csv(temp_file)
            all_records = existing.to_dict('records')
            print(f"[{species_name}] Resuming from {offset:,} records...")
    
    # Get total count for progress bar
    total_count = get_total_count(species_name, year_from)
    if total_count is None:
        print(f"[{species_name}] Could not determine total count")
        total_count = 0
    else:
        print(f"[{species_name}] Total available: {total_count:,} records")
    
    # Create progress bar
    pbar = tqdm(
        total=total_count if total_count > 0 else None,
        desc=f"{species_name[:30]}",
        unit="records",
        initial=offset
    )
    
    try:
        while True:
            try:
                batch = occ.search(
                    scientificName=species_name,
                    country=['US', 'CA', 'MX'],
                    hasCoordinate=True,
                    hasGeospatialIssue=False,
                    year=f'{year_from},2025',
                    limit=300,
                    offset=offset
                )
                
                if not batch['results']:
                    break
                
                # Filter for required fields
                filtered_results = [
                    r for r in batch['results']
                    if r.get('decimalLatitude') is not None 
                    and r.get('decimalLongitude') is not None
                    and r.get('eventDate') is not None
                ]
                
                all_records.extend(filtered_results)
                offset += 300
                pbar.update(len(filtered_results))
                
                # Save checkpoint every 1000 records
                if offset % 1000 == 0:
                    pd.DataFrame(all_records).to_csv(temp_file, index=False)
                    save_checkpoint(species_dir, offset, len(all_records))
                
                time.sleep(0.1)
                    
            except Exception as e:
                pbar.write(f"[{species_name}] Error at offset {offset}: {e}")
                pd.DataFrame(all_records).to_csv(temp_file, index=False)
                save_checkpoint(species_dir, offset, len(all_records))
                raise
        
        pbar.close()
        
        # Final processing
        df = pd.DataFrame(all_records)
        
        # Validate year
        if not df.empty and 'eventDate' in df.columns:
            df['year'] = pd.to_datetime(df['eventDate'], errors='coerce').dt.year
            df = df[df['year'] >= year_from]
            df = df.drop('year', axis=1)
        
        # Save final file
        df.to_csv(output_file, index=False)
        print(f"[{species_name}] Complete: {len(df):,} records → {output_file}")
        
        # Cleanup temp files
        if temp_file.exists():
            temp_file.unlink()
        checkpoint_file = species_dir / '.checkpoint.json'
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        return {'species': species_name, 'records': len(df), 'success': True}
        
    except KeyboardInterrupt:
        pbar.close()
        print(f"\n[{species_name}] Paused. Run again to resume from offset {offset}")
        raise
    except Exception as e:
        pbar.close()
        print(f"[{species_name}] Error: {e}")
        return {'species': species_name, 'records': len(all_records), 'success': False}

# Occurrece data procurement function (multi-threaded)
def download_all_species(species_list, max_workers=3, year_from=2015):
    results = []
    
    # Get all counts first
    print("\nFetching total record counts...")
    counts = {}
    for species in species_list:
        count = get_total_count(species, year_from)
        counts[species] = count if count else 0
        if count:
            print(f"  {species}: {count:,} records")
    
    total_all = sum(counts.values())
    print(f"\nTotal records to download: {total_all:,}\n")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_species = {
            executor.submit(download_species, species, year_from): species 
            for species in species_list
        }
        
        for future in as_completed(future_to_species):
            species = future_to_species[future]
            try:
                result = future.result()
                results.append(result)
            except KeyboardInterrupt:
                print("\n\nDownload paused by user. Progress saved.")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                print(f"[{species}] FAILED: {e}")
                results.append({'species': species, 'records': 0, 'success': False})
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Download GBIF occurrence data for Lepidoptera species',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Download single species
  python procure.py --species "Danaus plexippus"
  
  # Download from file
  python procure.py --file species_list.txt
  
  # Adjust workers and year
  python procure.py -f species_list.txt -w 8 -y 2017
        '''
    )
    
    parser.add_argument(
        '--species', '-s',
        nargs='+',
        help='One or more species names'
    )
    
    parser.add_argument(
        '--file', '-f',
        help='Path to text file with species names (one per line)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='Number of concurrent downloads (default: 8)'
    )
    
    parser.add_argument(
        '--year', '-y',
        type=int,
        default=2015,
        help='Download records from this year onwards (default: 2015)'
    )
    
    args = parser.parse_args()
    
    # Determine species list
    species_list = []
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' not found")
            return
        
        with open(args.file, 'r') as f:
            species_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(species_list)} species from {args.file}")
    
    elif args.species:
        species_list = args.species
    
    else:
        print("Error: Please provide species via --species or --file")
        parser.print_help()
        return
    
    if not species_list:
        print("Error: No species to download")
        return
    
    # Display configuration
    print("\n" + "="*50)
    print(f"DOWNLOADING {len(species_list)} SPECIES")
    print(f"Year range: {args.year}-2025")
    print(f"Concurrent workers: {args.workers}")
    print("="*50 + "\n")
    
    try:
        # Run downloads
        results = download_all_species(
            species_list, 
            max_workers=args.workers, 
            year_from=args.year
        )
        
        # Print summary
        print("\n" + "="*50)
        print("DOWNLOAD SUMMARY")
        print("="*50)
        successful = 0
        total_records = 0
        for r in results:
            status = "✓" if r['success'] else "✗"
            print(f"{status} {r['species']}: {r['records']:,} records")
            if r['success']:
                successful += 1
                total_records += r['records']
        
        print("="*50)
        print(f"Success: {successful}/{len(species_list)} species")
        print(f"Total records: {total_records:,}")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\n\nDownload interrupted. Re-run to resume.")


if __name__ == '__main__':
    main()

# Usage example
# python3 procure.py --s 'Danaus plexippus' --w 10