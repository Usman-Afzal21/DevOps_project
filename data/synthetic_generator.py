import pandas as pd
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Configuration
branches = [f"Branch_{i}" for i in range(1, 6)]
num_weeks = 520  # 1 year of weekly data
start_date = datetime(2024, 1, 1)

# Helper to generate alerts based on conditions
def generate_alert(row):
    alerts = []
    if row['revenue_change_pct'] <= -20:
        alerts.append("Significant revenue drop")
    if row['complaints'] > 30:
        alerts.append("Spike in customer complaints")
    if row['atm_uptime_percent'] < 92:
        alerts.append("ATM downtime increased")
    if row['fraud_flags'] > 0:
        alerts.append("Potential fraud detected")
    if not alerts:
        return "No significant issues"
    return "; ".join(alerts)

# Generate dataset
data = []
for branch in branches:
    last_week_revenue = random.uniform(80000, 120000)
    for week in range(num_weeks):
        date = start_date + timedelta(weeks=week)
        revenue = round(random.uniform(50000, 150000), 2)
        transactions = random.randint(1000, 5000)
        complaints = random.randint(0, 50)
        atm_uptime = round(random.uniform(89, 100), 2)
        staff_count = random.randint(10, 50)
        fraud_flags = random.choice([0]*9 + [1])
        revenue_change_pct = round(((revenue - last_week_revenue) / last_week_revenue) * 100, 2)
        last_week_revenue = revenue

        row = {
            "date": date,
            "branch": branch,
            "revenue": revenue,
            "transactions": transactions,
            "complaints": complaints,
            "atm_uptime_percent": atm_uptime,
            "staff_count": staff_count,
            "fraud_flags": fraud_flags,
            "revenue_change_pct": revenue_change_pct,
        }
        row["alert_summary"] = generate_alert(row)
        data.append(row)

# Save as DataFrame and CSV
df_mis_alerts = pd.DataFrame(data)
df_mis_alerts.to_csv("synthetic_mis_alerts.csv", index=False)

print("✅ Dataset saved as synthetic_mis_alerts.csv")
