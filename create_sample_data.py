import pandas as pd
import numpy as np
import datetime
import random

# Set seed for reproducibility
np.random.seed(42)

# Create sample transaction data with only 15 entries that will trigger alerts
n_transactions = 15
start_date = datetime.datetime(2024, 5, 1)
end_date = datetime.datetime(2024, 5, 10)

# Generate random dates
date_range = (end_date - start_date).days
transaction_dates = [start_date + datetime.timedelta(days=random.randint(0, date_range)) for _ in range(n_transactions)]
transaction_dates.sort()

# Transaction types
transaction_types = ['Debit', 'Credit', 'Transfer', 'ATM Withdrawal']

# Locations/Branches
locations = ['New York', 'San Francisco', 'Chicago', 'Houston', 'Dallas']

# Categories
categories = ['Groceries', 'Shopping', 'Dining', 'Entertainment', 'Travel']

# Customer IDs - use a small set to create patterns
customer_ids = [f'CUST{id}' for id in range(1001, 1006)]

# Generate base data
data = {
    'TransactionID': [f'TX{str(i+1000).zfill(6)}' for i in range(n_transactions)],
    'Date': transaction_dates,
    'TransactionAmount': np.round(np.random.normal(100, 30, n_transactions), 2),
    'TransactionType': [random.choice(transaction_types) for _ in range(n_transactions)],
    'CustomerID': [random.choice(customer_ids) for _ in range(n_transactions)],
    'Location': [random.choice(locations) for _ in range(n_transactions)],
    'Category': [random.choice(categories) for _ in range(n_transactions)],
    'Description': [f'{random.choice(categories)} purchase' for _ in range(n_transactions)],
    'AccountBalance': np.round(np.random.normal(2000, 500, n_transactions), 2),
    'LoginAttempts': [1] * n_transactions,
    'FraudFlag': [False] * n_transactions,
    'Channel': [random.choice(['Online', 'Branch', 'Mobile', 'ATM', 'Phone']) for _ in range(n_transactions)]
}

# Add pattern of transactions for a single customer to trigger unusual activity alert
customer_pattern_idx = [0, 1, 2, 3]
for idx in customer_pattern_idx:
    data['CustomerID'][idx] = 'CUST1001'
    data['TransactionType'][idx] = 'Debit'
    data['Date'][idx] = start_date + datetime.timedelta(hours=idx*3)

# 1. Add large transactions to trigger amount-based alerts (outliers)
large_tx_indices = [4, 5, 6]
for idx in large_tx_indices:
    data['TransactionAmount'][idx] = np.round(random.uniform(500, 1000), 2)
    data['TransactionType'][idx] = 'Debit'

# 2. Add transactions with high login attempts to trigger security alerts
high_login_indices = [7, 8, 9]
for idx in high_login_indices:
    data['LoginAttempts'][idx] = random.randint(5, 10)

# 3. Add explicit fraud flagged transactions
fraud_indices = [10, 11, 12]
for idx in fraud_indices:
    data['FraudFlag'][idx] = True
    # Also set some properties typical of fraudulent transactions
    data['TransactionAmount'][idx] = np.round(random.uniform(300, 800), 2)
    data['Location'][idx] = random.choice(['Miami', 'Las Vegas', 'Seattle'])  # Different locations than usual

# 4. Add unusual time pattern transactions (nighttime)
unusual_time_idx = [13, 14]
for idx in unusual_time_idx:
    # Set to early morning hours
    data['Date'][idx] = start_date + datetime.timedelta(days=idx) + datetime.timedelta(hours=random.randint(1, 4))
    data['CustomerID'][idx] = 'CUST1002'  # Same customer

# Create DataFrame
df = pd.DataFrame(data)

# Add CustomerAge and TransactionDuration columns which are needed by the system
df['CustomerAge'] = np.random.randint(18, 75, size=n_transactions)
df['TransactionDuration'] = np.random.randint(5, 120, size=n_transactions)

# Add a calculated PreviousTransactionDate column
df['PreviousTransactionDate'] = df['Date'] - pd.Timedelta(days=random.randint(1, 10))

# Output file
output_file = 'test_alert_transactions.csv'
df.to_csv(output_file, index=False)

print(f'Created test transaction data with {n_transactions} records: {output_file}')
print(f'Alert triggers included:')
print(f'- {len(fraud_indices)} explicitly fraud-flagged transactions')
print(f'- {len(large_tx_indices)} unusually large transactions')
print(f'- {len(high_login_indices)} transactions with high login attempts')
print(f'- {len(unusual_time_idx)} transactions at unusual times')
print(f'- {len(customer_pattern_idx)} transactions in quick succession for one customer')
print(f'\nPreview of the data:')
print(df.head().to_string()) 