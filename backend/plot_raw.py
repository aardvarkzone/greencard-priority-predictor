import pandas as pd
import matplotlib.pyplot as plt

# Read the processed data
emp_df = pd.read_csv('data/processed/processed_employment.csv')
fam_df = pd.read_csv('data/processed/processed_family.csv')

# Convert date to datetime
emp_df['date'] = pd.to_datetime(emp_df['date'])
fam_df['date'] = pd.to_datetime(fam_df['date'])

# Create employment-based plot
plt.figure(figsize=(15, 7))
plt.plot(emp_df['date'], emp_df['eb1_india_days'], label='EB1')
plt.plot(emp_df['date'], emp_df['eb2_india_days'], label='EB2')
plt.plot(emp_df['date'], emp_df['eb3_india_days'], label='EB3')

plt.title('Employment-Based Priority Dates Movement (India)')
plt.xlabel('Date')
plt.ylabel('Days from Reference')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/processed/employment_trends.png')

# Create family-based plot
plt.figure(figsize=(15, 7))
plt.plot(fam_df['date'], fam_df['f1_india_days'], label='F1')
plt.plot(fam_df['date'], fam_df['f2a_india_days'], label='F2A')
plt.plot(fam_df['date'], fam_df['f2b_india_days'], label='F2B')
plt.plot(fam_df['date'], fam_df['f3_india_days'], label='F3')
plt.plot(fam_df['date'], fam_df['f4_india_days'], label='F4')

plt.title('Family-Based Priority Dates Movement (India)')
plt.xlabel('Date')
plt.ylabel('Days from Reference')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('data/processed/family_trends.png')

plt.show()