import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Create a dataframe from scratch instead of reading from existing file
# Define the number of transactions to create
total_transactions = 30

# Create empty dataframe with all required columns
df = pd.DataFrame({
    'TransactionID': [f"TX{str(i+1000).zfill(6)}" for i in range(total_transactions)],
    'CustomerID': [f"CUST{1000+i}" for i in range(total_transactions)],
    'TransactionAmount': [random.uniform(50, 500) for _ in range(total_transactions)],
    'TransactionType': np.random.choice(['Debit', 'Credit', 'Transfer', 'ATM Withdrawal', 'Deposit'], size=total_transactions),
    'Location': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Houston', 'Dallas'], size=total_transactions),
    'Date': [(datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S") for _ in range(total_transactions)],
    'Channel': np.random.choice(['Online', 'Branch', 'Mobile', 'ATM', 'Phone'], size=total_transactions),
    'FraudFlag': [False] * total_transactions,
    'LoginAttempts': [random.randint(1, 3) for _ in range(total_transactions)],
    'AccountBalance': [random.uniform(1000, 10000) for _ in range(total_transactions)],
    'Description': ['Standard transaction'] * total_transactions,
    'TransactionDuration': [random.randint(30, 180) for _ in range(total_transactions)],
    'Category': np.random.choice(['Groceries', 'Shopping', 'Dining', 'Entertainment', 'Travel'], size=total_transactions),
    'PreviousTransactionDate': [(datetime.now() - timedelta(days=random.randint(31, 60))).strftime("%Y-%m-%d %H:%M:%S") for _ in range(total_transactions)]
})

# ==== ANOMALY PATTERN 1: Suspicious Time Patterns ====
# Create late-night transactions (1am-4am) for customer CUST1001
night_indices = list(range(0, 3))
for idx in night_indices:
    base_date = datetime.strptime("2024-05-10", "%Y-%m-%d")
    night_hour = random.randint(1, 4)
    df.loc[idx, 'Date'] = (base_date + timedelta(days=idx, hours=night_hour)).strftime("%Y-%m-%d %H:%M:%S")
    df.loc[idx, 'CustomerID'] = 'CUST1001'
    df.loc[idx, 'Channel'] = 'Online'
    df.loc[idx, 'TransactionType'] = 'Transfer'
    df.loc[idx, 'FraudFlag'] = True
    df.loc[idx, 'Description'] = 'Suspicious nighttime activity'

# ==== ANOMALY PATTERN 2: High-Velocity Transactions ====
# Create multiple transactions within minutes for the same customer
velocity_indices = list(range(3, 7))
for i, idx in enumerate(velocity_indices):
    base_time = datetime.strptime("2024-05-15 14:00:00", "%Y-%m-%d %H:%M:%S")
    df.loc[idx, 'Date'] = (base_time + timedelta(minutes=i*5)).strftime("%Y-%m-%d %H:%M:%S")
    df.loc[idx, 'CustomerID'] = 'CUST2000'
    df.loc[idx, 'TransactionAmount'] = random.uniform(50, 150)
    df.loc[idx, 'Channel'] = 'Mobile'
    df.loc[idx, 'Location'] = 'Las Vegas'
    df.loc[idx, 'Description'] = 'High-velocity transaction pattern'

# ==== ANOMALY PATTERN 3: Structuring Behavior ====
# Create multiple transactions just under reporting thresholds
structuring_indices = list(range(7, 12))
for idx in structuring_indices:
    df.loc[idx, 'Date'] = f"2024-05-{16+idx-7} 10:00:00"
    df.loc[idx, 'TransactionAmount'] = random.uniform(9750, 9950)  # Just under 10k
    df.loc[idx, 'CustomerID'] = 'CUST3000'
    df.loc[idx, 'Channel'] = 'Branch'
    df.loc[idx, 'TransactionType'] = 'Cash Withdrawal'
    df.loc[idx, 'Location'] = 'Miami'
    df.loc[idx, 'Description'] = 'Possible structuring behavior'

# ==== ANOMALY PATTERN 4: Round Amount Transactions ====
# Create suspicious round-amount transactions
round_amount_indices = list(range(12, 15))
for idx in round_amount_indices:
    df.loc[idx, 'Date'] = f"2024-05-{20+idx-12} 16:00:00"
    df.loc[idx, 'TransactionAmount'] = float(random.choice([1000, 2000, 3000, 5000, 10000]))
    df.loc[idx, 'CustomerID'] = 'CUST4000'
    df.loc[idx, 'Channel'] = 'ATM'
    df.loc[idx, 'Location'] = 'New York'
    df.loc[idx, 'Description'] = 'Suspicious round amount'

# ==== ANOMALY PATTERN 5: Unusual Location Sequence ====
# Create transactions showing unusual customer movement
location_seq_indices = list(range(15, 18))
unusual_location_sequence = ['New York', 'Los Angeles', 'Miami']
for i, idx in enumerate(location_seq_indices):
    df.loc[idx, 'Date'] = f"2024-05-{22} {i*6:02d}:00:00"  # Same day, different hours
    df.loc[idx, 'CustomerID'] = 'CUST5000'
    df.loc[idx, 'Location'] = unusual_location_sequence[i]
    df.loc[idx, 'Channel'] = 'ATM'
    df.loc[idx, 'Description'] = 'Unusual geographic movement'

# ==== ANOMALY PATTERN 6: New Account High Value ====
# Transactions with high values from new customers
new_account_indices = list(range(18, 21))
for idx in new_account_indices:
    df.loc[idx, 'Date'] = f"2024-05-{23+idx-18} 09:00:00"
    df.loc[idx, 'CustomerID'] = f'NEWCUST{6000+idx}'  # New customer IDs
    df.loc[idx, 'TransactionAmount'] = random.uniform(15000, 25000)
    df.loc[idx, 'TransactionType'] = 'Deposit'
    df.loc[idx, 'Channel'] = 'Branch'
    df.loc[idx, 'Description'] = 'New account with large initial deposit'

# ==== ANOMALY PATTERN 7: Login/Logout Anomalies ====
# Create transactions with extreme login attempts or durations
login_anomaly_indices = list(range(21, 24))
for idx in login_anomaly_indices:
    df.loc[idx, 'Date'] = f"2024-05-{24} {idx-20:02d}:00:00"
    df.loc[idx, 'CustomerID'] = 'CUST7000'
    df.loc[idx, 'LoginAttempts'] = random.randint(30, 50)  # Extremely high
    df.loc[idx, 'Channel'] = 'Online'
    df.loc[idx, 'Description'] = 'Suspicious login activity'
    df.loc[idx, 'TransactionDuration'] = random.randint(300, 600)  # Unusually long session

# ==== ANOMALY PATTERN 8: Cross-Border Activity ====
# Create international transactions with high values
international_indices = list(range(24, 27))
international_locations = ['Dubai', 'Hong Kong', 'Moscow', 'Zürich', 'Cayman Islands']
for idx in international_indices:
    df.loc[idx, 'Date'] = f"2024-05-25 {idx-23:02d}:00:00"
    df.loc[idx, 'CustomerID'] = 'CUST8000'
    df.loc[idx, 'TransactionAmount'] = random.uniform(50000, 100000)
    df.loc[idx, 'Location'] = random.choice(international_locations)
    df.loc[idx, 'TransactionType'] = 'Wire Transfer'
    df.loc[idx, 'Channel'] = 'Branch'
    df.loc[idx, 'FraudFlag'] = True
    df.loc[idx, 'Description'] = 'High-value international wire transfer'

# ==== ANOMALY PATTERN 9: Account Takeover Patterns ====
# Create transactions showing typical account takeover pattern
# First small test transactions, then large withdrawals
takeover_indices = list(range(27, 30))
for i, idx in enumerate(takeover_indices):
    df.loc[idx, 'Date'] = f"2024-05-{26} {12+i*2:02d}:00:00"
    df.loc[idx, 'CustomerID'] = 'CUST9000'
    df.loc[idx, 'Channel'] = 'Online'
    # First small test transactions, then large withdrawal
    if i < 2:
        df.loc[idx, 'TransactionAmount'] = random.uniform(1, 5)
        df.loc[idx, 'Description'] = 'Small test transaction'
    else:
        df.loc[idx, 'TransactionAmount'] = random.uniform(8000, 15000)
        df.loc[idx, 'Description'] = 'Large withdrawal after test transactions'
        df.loc[idx, 'FraudFlag'] = True

# Set transaction IDs with unique prefix to make them easily identifiable
df['TransactionID'] = [f"ALERT{str(i+1000).zfill(6)}" for i in range(len(df))]

# Make sure all fraud flags are normalized to boolean
df['FraudFlag'] = df['FraudFlag'].fillna(False).map({True: True, False: False, 'True': True, 'False': False, 1: True, 0: False})

# Save enhanced test data
output_file = 'enhanced_test_data.csv'
df.to_csv(output_file, index=False)

print(f"Enhanced test data saved to {output_file}")
print(f"Created {len(df)} test transactions with the following anomaly patterns:")
print("1. Suspicious nighttime transactions (1am-4am)")
print("2. High-velocity transactions (multiple transactions within minutes)")
print("3. Structuring behavior (multiple transactions just under reporting thresholds)")
print("4. Suspicious round amount transactions")
print("5. Unusual location sequence (improbable customer movement)")
print("6. New accounts with high-value transactions")
print("7. Login anomalies (extremely high login attempts)")
print("8. Cross-border high-value transfers")
print("9. Account takeover pattern (small test transactions followed by large withdrawal)")
print("\nAll transactions have ALERT prefix in their IDs for easy identification") 