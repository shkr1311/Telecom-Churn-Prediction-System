"""
Generate Large Telecom Churn Dataset
Run this script to create 4 CSV files with 1000+ customers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
NUM_CUSTOMERS = 1000

# ============================================================================
# GENERATE CUSTOMERS DATA
# ============================================================================

print("Generating Customers data...")

indian_first_names = [
    'Aarav', 'Vivaan', 'Aditya', 'Arjun', 'Sai', 'Ayaan', 'Krishna', 'Ishaan', 'Shaurya', 'Atharv',
    'Aadhya', 'Ananya', 'Diya', 'Ira', 'Prisha', 'Saanvi', 'Sara', 'Kiara', 'Aarohi', 'Navya',
    'Ravi', 'Amit', 'Rohan', 'Rahul', 'Karan', 'Vikram', 'Manoj', 'Suresh', 'Rajesh', 'Ankit',
    'Priya', 'Neha', 'Pooja', 'Sneha', 'Divya', 'Anjali', 'Shreya', 'Ritu', 'Kavita', 'Preeti',
    'Aryan', 'Dev', 'Kabir', 'Vihaan', 'Advait', 'Reyansh', 'Aayush', 'Dhruv', 'Arnav', 'Shivansh',
    'Anaya', 'Myra', 'Anika', 'Riya', 'Avni', 'Ishita', 'Nisha', 'Meera', 'Tara', 'Zara'
]

indian_last_names = [
    'Sharma', 'Verma', 'Patel', 'Kumar', 'Singh', 'Gupta', 'Reddy', 'Rao', 'Nair', 'Iyer',
    'Mehta', 'Shah', 'Jain', 'Das', 'Desai', 'Bose', 'Kapoor', 'Malhotra', 'Khanna', 'Chopra',
    'Agarwal', 'Banerjee', 'Chatterjee', 'Mukherjee', 'Saxena', 'Mishra', 'Pandey', 'Joshi',
    'Kulkarni', 'Deshpande', 'Bhatt', 'Trivedi', 'Menon', 'Krishnan', 'Pillai', 'Naidu',
    'Sinha', 'Ghosh', 'Roy', 'Choudhury', 'Patil', 'Jadhav', 'More', 'Pawar', 'Kaur', 'Dhillon'
]

plan_types = ['Prepaid', 'Postpaid']
regions = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Pune', 'Bengaluru', 'Hyderabad', 
           'Jaipur', 'Chandigarh', 'Ahmedabad', 'Lucknow', 'Kochi', 'Indore', 'Nagpur', 'Surat']

customers_data = []
for i in range(1, NUM_CUSTOMERS + 1):
    customer_id = 1000 + i
    first_name = random.choice(indian_first_names)
    last_name = random.choice(indian_last_names)
    name = f"{first_name} {last_name}"
    plan_type = random.choice(plan_types)
    region = random.choice(regions)
    
    customers_data.append({
        'customer_id': customer_id,
        'name': name,
        'plan_type': plan_type,
        'region': region
    })

customers_df = pd.DataFrame(customers_data)

# ============================================================================
# GENERATE BILLING DATA
# ============================================================================

print("Generating Billing data...")

contract_types = ['Month-to-Month', 'One Year', 'Two Year']

billing_data = []
for i in range(1, NUM_CUSTOMERS + 1):
    customer_id = 1000 + i
    plan_type = customers_df[customers_df['customer_id'] == customer_id]['plan_type'].values[0]
    
    # Tenure: Month-to-Month tends to be shorter
    contract_type = random.choice(contract_types)
    if contract_type == 'Month-to-Month':
        tenure = np.random.randint(1, 25)  # Shorter tenure
        churn_prob = 0.6  # Higher churn probability
    elif contract_type == 'One Year':
        tenure = np.random.randint(6, 36)
        churn_prob = 0.3
    else:  # Two Year
        tenure = np.random.randint(12, 60)
        churn_prob = 0.15
    
    # Monthly charges based on plan type
    if plan_type == 'Prepaid':
        base_charge = np.random.randint(100, 400)
    else:  # Postpaid
        base_charge = np.random.randint(200, 800)
    
    monthly_charges = base_charge + np.random.randint(-50, 100)
    
    # Higher charges increase churn probability slightly
    if monthly_charges > 500:
        churn_prob += 0.15
    
    # Determine churn
    churn = 'Yes' if random.random() < churn_prob else 'No'
    
    billing_data.append({
        'customer_id': customer_id,
        'tenure': tenure,
        'contract_type': contract_type,
        'monthly_charges': monthly_charges,
        'churn': churn
    })

billing_df = pd.DataFrame(billing_data)

# ============================================================================
# GENERATE USAGE DATA
# ============================================================================

print("Generating Usage data...")

usage_data = []
for i in range(1, NUM_CUSTOMERS + 1):
    customer_id = 1000 + i
    plan_type = customers_df[customers_df['customer_id'] == customer_id]['plan_type'].values[0]
    monthly_charges = billing_df[billing_df['customer_id'] == customer_id]['monthly_charges'].values[0]
    
    # Data usage correlates with plan type and charges
    if plan_type == 'Prepaid':
        data_used_gb = round(np.random.uniform(1, 30), 2)
        calls_made = np.random.randint(10, 100)
    else:  # Postpaid
        data_used_gb = round(np.random.uniform(5, 80), 2)
        calls_made = np.random.randint(20, 200)
    
    # Revenue roughly correlates with monthly charges
    revenue_inr = monthly_charges + np.random.randint(-50, 150)
    if revenue_inr < 100:
        revenue_inr = 100
    
    usage_data.append({
        'customer_id': customer_id,
        'data_used_gb': data_used_gb,
        'calls_made': calls_made,
        'revenue_inr': revenue_inr
    })

usage_df = pd.DataFrame(usage_data)

# ============================================================================
# GENERATE COMPLAINTS DATA
# ============================================================================

print("Generating Complaints data...")

complaint_categories = ['Billing', 'Network', 'Support', 'Recharge', 'Data', 'Call Quality', 'Service']
complaint_statuses = ['Open', 'Closed']

# Generate start date (6 months ago)
start_date = datetime.now() - timedelta(days=180)

complaints_data = []

# About 40% of customers have complaints
num_customers_with_complaints = int(NUM_CUSTOMERS * 0.4)
customers_with_complaints = random.sample(range(1001, 1001 + NUM_CUSTOMERS), num_customers_with_complaints)

for customer_id in customers_with_complaints:
    # Each complaining customer has 1-5 complaints
    num_complaints = random.randint(1, 5)
    
    for _ in range(num_complaints):
        # Random date within last 6 months
        days_ago = random.randint(0, 180)
        complaint_date = start_date + timedelta(days=days_ago)
        
        category = random.choice(complaint_categories)
        
        # Older complaints more likely to be closed
        if days_ago > 30:
            status = random.choice(['Open', 'Closed', 'Closed', 'Closed'])  # 75% closed
        else:
            status = random.choice(['Open', 'Open', 'Closed'])  # 33% closed
        
        created_at = complaint_date.strftime('%Y-%m-%d %H:%M')
        
        complaints_data.append({
            'customer_id': customer_id,
            'category': category,
            'created_at': created_at,
            'status': status
        })

complaints_df = pd.DataFrame(complaints_data)

# ============================================================================
# SAVE FILES
# ============================================================================

print("\nSaving CSV files...")

customers_df.to_csv('customers.csv', index=False)
print(f"✓ customers.csv saved ({len(customers_df)} rows)")

billing_df.to_csv('billing.csv', index=False)
print(f"✓ billing.csv saved ({len(billing_df)} rows)")

usage_df.to_csv('usage_data.csv', index=False)
print(f"✓ usage_data.csv saved ({len(usage_df)} rows)")

complaints_df.to_csv('complaints.csv', index=False)
print(f"✓ complaints.csv saved ({len(complaints_df)} rows)")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*60)
print("DATASET SUMMARY")
print("="*60)

print(f"\n📊 CUSTOMERS ({len(customers_df)} total)")
print(customers_df['plan_type'].value_counts())
print("\nTop 5 Regions:")
print(customers_df['region'].value_counts().head())

print(f"\n💰 BILLING ({len(billing_df)} total)")
print(billing_df['churn'].value_counts())
print(f"\nChurn Rate: {(billing_df['churn'] == 'Yes').sum() / len(billing_df) * 100:.2f}%")
print("\nContract Types:")
print(billing_df['contract_type'].value_counts())
print(f"\nAverage Monthly Charges: ₹{billing_df['monthly_charges'].mean():.2f}")
print(f"Average Tenure: {billing_df['tenure'].mean():.1f} months")

print(f"\n📱 USAGE ({len(usage_df)} total)")
print(f"Average Data Used: {usage_df['data_used_gb'].mean():.2f} GB")
print(f"Average Calls Made: {usage_df['calls_made'].mean():.0f}")
print(f"Average Revenue: ₹{usage_df['revenue_inr'].mean():.2f}")

print(f"\n📞 COMPLAINTS ({len(complaints_df)} total)")
print(f"Customers with complaints: {len(customers_with_complaints)} ({len(customers_with_complaints)/NUM_CUSTOMERS*100:.1f}%)")
print("\nComplaint Categories:")
print(complaints_df['category'].value_counts())
print("\nComplaint Status:")
print(complaints_df['status'].value_counts())

print("\n" + "="*60)
print("✅ All files generated successfully!")
print("="*60)
print("\nYou can now upload these files to the Streamlit app:")
print("  • customers.csv")
print("  • billing.csv")
print("  • usage_data.csv")
print("  • complaints.csv")