from pygbif import occurrences as occ
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import argparse
from tqdm import tqdm

# Paths
DATA_DIR = Path('occurrence-data')
SPECIES_LIST = DATA_DIR / 'species-list.txt'
QUEUE_FILE = DATA_DIR / 'queue.txt'
COMPLETED_FILE = DATA_DIR / 'completed.json'
FAILED_FILE = DATA_DIR / 'failed.json'

# Simple file helpers
def read_lines(file):
    return [line.strip() for line in open(file) if line.strip()] if file.exists() else []

def write_lines(file, lines):
    DATA_DIR.mkdir(exist_ok=True)
    file.write_text('\n'.join(lines) + '\n' if lines else '')

def read_json(file):
    return json.load(open(file)) if file.exists() else []

def write_json(file, data):
    DATA_DIR.mkdir(exist_ok=True)
    json.dump(data, open(file, 'w'), indent=2)

def log_result(species, status):
    """Remove from queue and log result"""
    queue = read_lines(QUEUE_FILE)
    write_lines(QUEUE_FILE, [s for s in queue if s != species])
    
    file = COMPLETED_FILE if status == 'complete' else FAILED_FILE
    log = read_json(file)
    log.append({'species': species, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
    write_json(file, log)

# Core download
def download_year(species, year, countries):
    """Download up to 100k records for one year with retry logic"""
    records = []
    offset = 0
    
    while offset < 100000:
        retries = 0
        max_retries = 5
        
        while retries < max_retries:
            try:
                batch = occ.search(
                    scientificName=species,
                    country=countries,
                    hasCoordinate=True,
                    hasGeospatialIssue=False,
                    year=str(year),
                    limit=min(300, 100000 - offset),
                    offset=offset
                )
                
                if not batch.get('results'):
                    return records
                
                records.extend([
                    r for r in batch['results']
                    if r.get('decimalLatitude') and r.get('decimalLongitude') and r.get('eventDate')
                ])
                
                offset += 300
                time.sleep(0.1)
                break  # Success, exit retry loop
                
            except Exception as e:
                if '429' in str(e):  # Rate limit
                    retries += 1
                    if retries < max_retries:
                        wait = 2 ** retries  # Exponential backoff: 2, 4, 8, 16 seconds
                        time.sleep(wait)
                    else:
                        return records  # Give up after max retries
                else:
                    return records  # Other error, stop this year
    
    return records

def download_species(species, year_from, countries):
    """Download all data for a species"""
    
    # Setup
    safe_name = species.replace(' ', '-').lower()
    species_dir = DATA_DIR / safe_name
    species_dir.mkdir(parents=True, exist_ok=True)
    output = species_dir / f'{safe_name}_gbif.csv'
    
    # Get count
    try:
        total = occ.search(
            scientificName=species,
            country=countries,
            hasCoordinate=True,
            year=f'{year_from},2025',
            limit=1
        ).get('count', 0)
    except:
        total = 0
    
    # Download (silent progress bar only)
    all_records = []
    pbar = tqdm(
        total=total, 
        desc=species[:30], 
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
        ncols=80
    )
    
    try:
        for year in range(year_from, 2026):
            year_data = download_year(species, year, countries)
            all_records.extend(year_data)
            pbar.update(len(year_data))
        
        pbar.close()
        
        # Save
        df = pd.DataFrame(all_records)
        df.to_csv(output, index=False)
        
        # Show completion with data quality info
        if total > 0 and len(df) < total * 0.9:
            print(f"[{species}] ⚠ {len(df):,}/{total:,} saved (some data may be missing)")
        else:
            print(f"[{species}] ✓ {len(df):,} saved")
        
        log_result(species, 'complete')
        return len(df)
        
    except Exception as e:
        pbar.close()
        print(f"[{species}] ✗ {e}")
        log_result(species, 'failed')
        return 0

def run_batch(year_from, countries, workers, force=False):
    """Process the species list"""
    
    # Load master list
    if not SPECIES_LIST.exists():
        print(f"Error: {SPECIES_LIST} not found")
        print("Create it with one species name per line")
        return
    
    master_list = read_lines(SPECIES_LIST)
    if not master_list:
        print(f"Error: {SPECIES_LIST} is empty")
        return
    
    # Initialize queue
    existing_queue = read_lines(QUEUE_FILE)
    queue = sorted(set(existing_queue + master_list))
    write_lines(QUEUE_FILE, queue)
    
    # Skip completed (unless force)
    completed = {item['species'] for item in read_json(COMPLETED_FILE)}
    
    if force:
        pending = queue
        print("🔄 Force mode: Re-downloading all species")
    else:
        pending = [s for s in queue if s not in completed]
    
    if not pending:
        print("✓ All species completed!")
        return
    
    # Show status
    print(f"\n{'='*60}")
    print(f"Species: {len(pending)} pending | {len(completed)} done | {len(read_json(FAILED_FILE))} failed")
    print(f"Settings: {year_from}-2025 | {', '.join(countries)} | {workers} workers")
    print(f"{'='*60}\n")
    
    # Download in parallel
    total_records = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_species, sp, year_from, countries): sp 
            for sp in pending
        }
        
        for future in as_completed(futures):
            try:
                records = future.result()
                if records > 0:
                    success_count += 1
                    total_records += records
            except KeyboardInterrupt:
                print("\n⏸ Paused. Run again to resume.")
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:
                sp = futures[future]
                print(f"[{sp}] ✗ {e}")
                log_result(sp, 'failed')
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✓ {success_count}/{len(pending)} successful | {total_records:,} total records")
    print(f"{'='*60}")

def show_status():
    """Show queue status"""
    pending = read_lines(QUEUE_FILE)
    completed = read_json(COMPLETED_FILE)
    failed = read_json(FAILED_FILE)
    
    print(f"\n{'='*60}")
    print("QUEUE STATUS")
    print(f"{'='*60}")
    
    print(f"\nPending ({len(pending)}):")
    for sp in pending:
        print(f"  • {sp}")
    
    print(f"\nCompleted ({len(completed)}):")
    for item in completed[-10:]:  # Last 10
        print(f"  ✓ {item['species']} - {item['time']}")
    if len(completed) > 10:
        print(f"  ... and {len(completed) - 10} more")
    
    print(f"\nFailed ({len(failed)}):")
    for item in failed:
        print(f"  ✗ {item['species']} - {item['time']}")
    
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description='Download GBIF data for species in occurrence-data/species-list.txt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run with defaults (uses species-list.txt)
  python procure.py
  
  # Custom settings
  python procure.py -w 5 -y 2010 -c US CA MX BR
  
  # Re-download everything with new year range
  python procure.py -y 2020 --force
  
  # Check status
  python procure.py --status
  
  # Retry failed
  python procure.py --retry
  
  # Clear and restart
  python procure.py --clear
        '''
    )
    
    parser.add_argument('-w', '--workers', type=int, default=3, help='Parallel downloads (default: 3, max recommended: 5)')
    parser.add_argument('-y', '--year', type=int, default=2015, help='Start year (default: 2015)')
    parser.add_argument('-c', '--countries', nargs='+', default=['US', 'CA', 'MX'], 
                       help='Country codes (default: US CA MX)')
    parser.add_argument('--status', action='store_true', help='Show queue status')
    parser.add_argument('--retry', action='store_true', help='Retry failed species')
    parser.add_argument('--clear', action='store_true', help='Clear queue/logs')
    parser.add_argument('--force', action='store_true', help='Re-download all species (ignores completed)')
    
    args = parser.parse_args()
    
    # Warn about too many workers
    if args.workers > 5:
        print(f"⚠️  Warning: {args.workers} workers may cause rate limiting (429 errors)")
        print("   Recommended: 3-5 workers for best performance\n")
        time.sleep(2)
    
    # Handle commands
    if args.status:
        show_status()
        return
    
    if args.clear:
        for f in [QUEUE_FILE, COMPLETED_FILE, FAILED_FILE]:
            f.unlink(missing_ok=True)
        print("✓ Cleared!")
        return
    
    if args.retry:
        failed = [item['species'] for item in read_json(FAILED_FILE)]
        if failed:
            FAILED_FILE.unlink(missing_ok=True)
            queue = read_lines(QUEUE_FILE)
            write_lines(QUEUE_FILE, sorted(set(queue + failed)))
            print(f"Added {len(failed)} species back to queue")
        else:
            print("No failed species to retry")
            return
    
    # Run
    try:
        run_batch(args.year, args.countries, args.workers, args.force)
    except KeyboardInterrupt:
        print("\n⏸ Interrupted")

if __name__ == '__main__':
    main()