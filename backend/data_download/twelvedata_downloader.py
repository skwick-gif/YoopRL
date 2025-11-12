"""
Twelve Data API Downloader for TNA and IWM
Downloads 15-minute candle data with much higher daily quota (800 calls)
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List
import urllib3
from pathlib import Path

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =============================================================================
# Configuration
# =============================================================================

API_KEY = "95bcfdd30f134cf89164c39bbce32181"  # Your Twelve Data API key
BASE_URL = "https://api.twelvedata.com"
SYMBOLS = ["IWM"]  # Symbols to download
INTERVAL = "15min"        # 15-minute intervals
MAX_CALLS_PER_DAY = 800   # Twelve Data free tier limit

# CSV files for each symbol
CSV_FILES = {
    "IWM": "IWM_15min_data_td.csv"
}
PROGRESS_FILE = "twelvedata_progress.json"

# =============================================================================
# Helper Functions
# =============================================================================

def create_month_list() -> List[str]:
    """
    Create list of months from 2025-11 down to 2020-01
    Returns list in format ['2025-11', '2025-10', ..., '2020-01']
    """
    months = []
    current_date = datetime(2025, 11, 1)  # Start from November 2025
    end_date = datetime(2020, 1, 1)       # Go back to January 2020
    
    while current_date >= end_date:
        month_str = current_date.strftime("%Y-%m")
        months.append(month_str)
        
        # Go to previous month
        if current_date.month == 1:
            current_date = current_date.replace(year=current_date.year - 1, month=12)
        else:
            current_date = current_date.replace(month=current_date.month - 1)
    
    return months

def load_progress() -> Dict:
    """Load download progress from JSON file"""
    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "downloaded": {},
            "current_day_calls": 0,
            "last_call_date": datetime.now().strftime("%Y-%m-%d")
        }

def save_progress(progress: Dict):
    """Save download progress to JSON file"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def reset_daily_counter_if_needed(progress: Dict):
    """Reset daily call counter if it's a new day"""
    today = datetime.now().strftime("%Y-%m-%d")
    if progress["last_call_date"] != today:
        progress["current_day_calls"] = 0
        progress["last_call_date"] = today
        save_progress(progress)
        print(f"🔄 New day detected - resetting call counter")

def download_data_twelvedata(symbol: str, month: str) -> Dict:
    """
    Download 15-minute data for a specific symbol and month from Twelve Data
    """
    # Calculate start and end dates for the month
    year, month_num = month.split('-')
    start_date = f"{year}-{month_num}-01"
    
    # Calculate last day of month
    if month_num == '12':
        next_month = f"{int(year)+1}-01-01"
    else:
        next_month = f"{year}-{int(month_num)+1:02d}-01"
    
    end_date = (datetime.strptime(next_month, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
    
    url = f"{BASE_URL}/time_series"
    params = {
        'symbol': symbol,
        'interval': INTERVAL,
        'start_date': start_date,
        'end_date': end_date,
        'apikey': API_KEY,
        'format': 'JSON',
        'outputsize': 5000  # Maximum output size
    }
    
    try:
        response = requests.get(url, params=params, verify=False, timeout=30)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Request failed: {e}")
        return {}
    except json.JSONDecodeError:
        print(f"  ❌ Invalid JSON response")
        return {}

def save_data_to_csv(symbol: str, month: str, data: Dict) -> bool:
    """
    Save data to unified CSV file, avoiding duplicates
    """
    if 'values' not in data or not data['values']:
        return False
    
    csv_file = CSV_FILES[symbol]
    
    # Convert API data to DataFrame
    records = []
    for item in data['values']:
        record = {
            'datetime': item['datetime'],
            'symbol': symbol,
            'month': month,
            'open': float(item['open']),
            'high': float(item['high']),
            'low': float(item['low']),
            'close': float(item['close']),
            'volume': int(item['volume']) if item['volume'] else 0
        }
        records.append(record)
    
    new_df = pd.DataFrame(records)
    
    # Load existing CSV or create new one
    if Path(csv_file).exists():
        existing_df = pd.read_csv(csv_file)
        
        # Remove any existing data for this month to avoid duplicates
        existing_df = existing_df[existing_df['month'] != month]
        
        # Combine dataframes
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    # Sort by datetime
    combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])
    combined_df = combined_df.sort_values('datetime')
    combined_df['datetime'] = combined_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    combined_df.to_csv(csv_file, index=False)
    
    return True

# =============================================================================
# Main Function
# =============================================================================

def main():
    print("🚀 Starting data download from Twelve Data")
    print(f"📊 Symbols: {', '.join(SYMBOLS)}")
    print(f"⏱️  Interval: {INTERVAL}")
    print(f"🎯 Daily quota: {MAX_CALLS_PER_DAY} calls")
    print("-" * 60)
    
    # Create month list
    months = create_month_list()
    total_combinations = len(months) * len(SYMBOLS)
    
    print(f"📅 Total {len(months)} months × {len(SYMBOLS)} symbols = {total_combinations} calls")
    print(f"⏳ Estimated time: {(total_combinations / MAX_CALLS_PER_DAY):.1f} days")
    print("-" * 60)
    
    # Load progress
    progress = load_progress()
    reset_daily_counter_if_needed(progress)
    
    successful_downloads = 0
    failed_downloads = 0
    
    try:
        # Download UPRO and SPY datasets
        for symbol in SYMBOLS:
            for month in months:
                
                # Check if already downloaded
                download_key = f"{symbol}_{month}"
                if download_key in progress["downloaded"]:
                    print(f"⏭️  Already downloaded: {symbol} {month}")
                    continue
                
                # Check daily quota
                if progress["current_day_calls"] >= MAX_CALLS_PER_DAY:
                    print(f"🛑 Reached daily quota ({MAX_CALLS_PER_DAY} calls)")
                    print("💤 Come back tomorrow to continue download")
                    break
                
                print(f"📥 [{progress['current_day_calls']+1}/{MAX_CALLS_PER_DAY}] {symbol} - {month}")
                print(f"  📡 Downloading {symbol} for {month}...")
                
                # Download data
                data = download_data_twelvedata(symbol, month)
                progress["current_day_calls"] += 1
                
                if data and 'values' in data and data['values']:
                    # Save to CSV
                    if save_data_to_csv(symbol, month, data):
                        print(f"  ✅ Success! {len(data['values'])} data points")
                        csv_file = CSV_FILES[symbol]
                        if Path(csv_file).exists():
                            total_rows = len(pd.read_csv(csv_file))
                            print(f"  💾 Updated file: {csv_file} (total {total_rows} rows)")
                        successful_downloads += 1
                        
                        # Mark as downloaded
                        progress["downloaded"][download_key] = True
                    else:
                        print(f"  ⚠️  Failed to save data for {symbol} {month}")
                        failed_downloads += 1
                else:
                    # Check for API error messages
                    if 'message' in data:
                        print(f"  ⚠️  API message: {data['message']}")
                    else:
                        print(f"  ⚠️  No time series data for {symbol} {month}")
                    
                    # Still mark as "downloaded" to avoid retrying
                    progress["downloaded"][download_key] = True
                    successful_downloads += 1
                
                print(f"  🎉 Total successful downloads: {successful_downloads}")
                
                # Save progress after each download
                save_progress(progress)
                
                # Rate limiting - 8 calls per minute = 7.5 seconds between calls
                print(f"  ⏳ Waiting 8 seconds...")
                time.sleep(8)
            
            # Break outer loop if quota reached
            if progress["current_day_calls"] >= MAX_CALLS_PER_DAY:
                break
                
    except KeyboardInterrupt:
        print(f"\n⛔ Stopped by user")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print(f"✅ Successful downloads: {successful_downloads}")
    print(f"❌ Failures: {failed_downloads}")
    print(f"🔄 Today's calls: {progress['current_day_calls']}/{MAX_CALLS_PER_DAY}")
    
    total_downloaded = len(progress["downloaded"])
    print(f"📈 Overall progress: {total_downloaded}/{total_combinations}")
    print(f"⏳ Remaining downloads: {total_combinations - total_downloaded}")
    
    if progress["current_day_calls"] >= MAX_CALLS_PER_DAY:
        print("💡 Run script tomorrow to continue!")
    else:
        print("💡 Ready for more downloads!")

if __name__ == "__main__":
    main()