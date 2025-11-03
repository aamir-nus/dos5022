"""
Download tender dataset from data.gov.sg

Dataset URL: https://data.gov.sg/datasets/d_acde1106003906a75c3fa052592f2fcb/view

Note: If running in a restricted environment, use generate_sample_data() to create sample data.
"""
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def generate_sample_data(n_samples=1000):
    """
    Generate sample tender data for development/testing
    Based on typical tender dataset structure from data.gov.sg
    """
    np.random.seed(42)
    
    # Sample agencies
    agencies = ['MOE', 'MOH', 'MND', 'MTI', 'MHA', 'MINDEF', 'MOT', 'MOM']
    
    # Sample categories
    categories = [
        'IT Services', 'Construction', 'Consultancy', 'Medical Supplies',
        'Security Services', 'Maintenance', 'Equipment', 'Training'
    ]
    
    # Sample suppliers
    suppliers = [f'Supplier_{i}' for i in range(1, 51)]
    
    # Generate data
    data = []
    start_date = datetime(2020, 1, 1)
    
    for i in range(n_samples):
        # Award status
        awarded = np.random.choice([True, False], p=[0.7, 0.3])
        
        # Generate fields
        tender_id = f"T{str(i+1).zfill(6)}"
        agency = np.random.choice(agencies)
        category = np.random.choice(categories)
        
        # Tender text - simulate description
        tender_text = f"{category} for {agency} - {np.random.choice(['Phase 1', 'Phase 2', 'Annual', 'Project'])}"
        
        # Published date
        days_offset = np.random.randint(0, 1460)  # 4 years
        published_date = (start_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        # Award amount (if awarded)
        if awarded:
            # Log-normal distribution for realistic award amounts
            base_amount = np.random.lognormal(10, 2)  # Mean around 22k
            award_amount = round(base_amount, 2)
            supplier = np.random.choice(suppliers)
            award_date = (datetime.strptime(published_date, '%Y-%m-%d') + 
                         timedelta(days=np.random.randint(30, 180))).strftime('%Y-%m-%d')
        else:
            award_amount = None
            supplier = None
            award_date = None
        
        data.append({
            'tender_id': tender_id,
            'agency': agency,
            'category': category,
            'tender_description': tender_text,
            'published_date': published_date,
            'award_status': 'Awarded' if awarded else 'No award',
            'award_amount': award_amount,
            'supplier': supplier,
            'award_date': award_date
        })
    
    return pd.DataFrame(data)

def download_dataset():
    """
    Download the tender dataset from data.gov.sg
    URL: https://data.gov.sg/datasets/d_acde1106003906a75c3fa052592f2fcb/view
    """
    
    # Dataset URL from data.gov.sg
    dataset_id = "d_acde1106003906a75c3fa052592f2fcb"
    url = f"https://data.gov.sg/api/action/datastore_search?resource_id={dataset_id}&limit=100000"
    
    print(f"Downloading data from {url}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        if 'result' in data and 'records' in data['result']:
            records = data['result']['records']
            df = pd.DataFrame(records)
            
            # Create data directory if it doesn't exist
            data_dir = Path(__file__).parent.parent / "data"
            data_dir.mkdir(exist_ok=True)
            
            # Save to CSV
            output_path = data_dir / "tender_data.csv"
            df.to_csv(output_path, index=False)
            
            print(f"Successfully downloaded {len(records)} records")
            print(f"Data saved to {output_path}")
            print(f"\nDataset shape: {df.shape}")
            print(f"\nColumns: {df.columns.tolist()}")
            
            return df
        else:
            print("Error: Unexpected data format")
            return None
            
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("\nGenerating sample data instead...")
        
        # Generate sample data
        df = generate_sample_data()
        
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)
        
        # Save to CSV
        output_path = data_dir / "tender_data.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Successfully generated {len(df)} sample records")
        print(f"Data saved to {output_path}")
        print(f"\nDataset shape: {df.shape}")
        print(f"\nColumns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        return df

if __name__ == "__main__":
    download_dataset()
