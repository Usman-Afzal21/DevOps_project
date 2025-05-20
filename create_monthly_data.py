import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_monthly_data(month, year):
    # Generate dates for the month
    start_date = datetime(year, month, 1)
    if month == 12:
        end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = datetime(year, month + 1, 1) - timedelta(days=1)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate random alert counts with some patterns
    np.random.seed(month)  # Different seed for each month
    base_alerts = np.random.randint(10, 30, size=len(dates))
    
    # Add some weekly patterns
    weekly_pattern = np.sin(np.arange(len(dates)) * 2 * np.pi / 7) * 5
    trend = np.linspace(0, 10, len(dates))  # Increasing trend
    
    alert_counts = base_alerts + weekly_pattern + trend
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'alert_count': alert_counts.astype(int)
    })
    
    return df

def main():
    # Create directory for monthly data
    data_dir = "data/monthly_alerts"
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data for January to May 2025
    for month in range(1, 6):
        df = generate_monthly_data(month, 2025)
        filename = f"{month}-25.csv"
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Generated {filename}")

if __name__ == "__main__":
    main() 