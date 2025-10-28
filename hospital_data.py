# -------------------------------------------------------
# Title: Hospital Patient Records Analysis
# Objective: Analyze hospital admission trends and patient recovery statistics
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set a clean style for the plots
sns.set_style("whitegrid")

# Define the file path
file_path = "hospital_data (1).csv"

# -------------------------------------------------------
# STEP 1: Load Dataset
# -------------------------------------------------------
print("\n==============================")
print(" HOSPITAL PATIENT RECORDS ANALYSIS ")
print("==============================\n")

try:
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully ✅\n")
    print("First 5 Rows:")
    print(df.head())
    print("\nColumns:", list(df.columns))
except FileNotFoundError:
    print(f"Error: '{file_path}' not found. Please ensure the file is in the correct path.")
    exit()

# -------------------------------------------------------
# STEP 2: Handle Missing recovery_days (Median Imputation)
# -------------------------------------------------------
print("\n===== STEP 2: Handling Missing Values (recovery_days) =====")
median_recovery = df['recovery_days'].median()
df['recovery_days'] = df['recovery_days'].fillna(median_recovery)

print(f"Median Recovery Days: {median_recovery:.2f}")
print("Missing recovery_days handled successfully using median imputation ✅")

# -------------------------------------------------------
# Q1: Average Recovery Days by Disease [CO1, BL3]
# -------------------------------------------------------
print("\n\n===== Q1: Average Recovery Days by Disease =====")
avg_recovery_by_disease = df.groupby('disease')['recovery_days'].mean().sort_values(ascending=False)
print(avg_recovery_by_disease.round(2))

# -------------------------------------------------------
# Q2: Analyze Age-wise Recovery Rate [CO2, BL4]
# -------------------------------------------------------
print("\n\n===== Q2: Age-wise Recovery Rate =====")

# Define age groups (bins) and labels
bins = [0, 18, 35, 50, 65, 100]
labels = ['0-18', '19-35', '36-50', '51-65', '66+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

agewise_recovery = df.groupby('age_group', observed=False)['recovery_days'].mean()
print(agewise_recovery.round(2))

# -------------------------------------------------------
# Q3: Replace Missing Gender Values with 'Unknown' [CO3, BL3]
# -------------------------------------------------------
print("\n\n===== Q3: Replacing Missing Gender Values =====")
missing_before = df['gender'].isna().sum()
df['gender'] = df['gender'].fillna('Unknown')
missing_after = df['gender'].isna().sum()

print(f"Missing Genders Before: {missing_before}")
print(f"Missing Genders After Replacement: {missing_after}")
print("All missing gender values replaced with 'Unknown' ✅")

# -------------------------------------------------------
# Q4: Identify Peak Hospitalization Periods by Month [CO4, BL4]
# -------------------------------------------------------
print("\n\n===== Q4: Peak Hospitalization Periods by Month =====")

df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
df['admission_month'] = df['admission_date'].dt.month_name()

month_order = [
    'January', 'February', 'March', 'April', 'May', 'June', 
    'July', 'August', 'September', 'October', 'November', 'December'
]
monthwise_count = df['admission_month'].value_counts().reindex(month_order).dropna()
print(monthwise_count.astype(int))

# -------------------------------------------------------
# Q5: Visualize Recovery Trends (Bar & Grouped Bar) [CO5, BL5]
# -------------------------------------------------------
print("\n\n===== Q5: Visualizing Recovery Trends (Bar & Grouped Bar) =====")

# --- Plot 1: Average Recovery Days by Disease (Bar Plot) ---
plt.figure(figsize=(10, 6))
sns.barplot(
    x=avg_recovery_by_disease.index,
    y=avg_recovery_by_disease.values,
    hue=avg_recovery_by_disease.index,
    palette='Set1' # Changed to a lighter blue-green palette
)
plt.title("Plot 1: Average Recovery Days by Disease", fontsize=14, fontweight='bold')
plt.xlabel("Disease", fontsize=12)
plt.ylabel("Average Recovery Days", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
if plt.gca().get_legend() is not None:
    plt.gca().get_legend().remove()
plt.show()


# --- Plot 2: Average Recovery Days by Age Group and Gender (Grouped Bar Plot) ---

pivot_table = df.pivot_table(
    values='recovery_days',
    index='age_group',
    columns='gender',
    aggfunc='mean',
    observed=False
)

print("\nAverage Recovery Days (Age Group vs Gender):")
print(pivot_table.round(2))

# Plotting the grouped bar chart
pivot_table.plot(kind='bar', figsize=(10, 6), width=0.8, cmap='Paired') # Changed colormap to 'Paired'

plt.title("Plot 2: Average Recovery Days by Age Group and Gender (Grouped Bar)", fontsize=14, fontweight='bold')
plt.xlabel("Age Group", fontsize=12)
plt.ylabel("Average Recovery Days", fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Gender')
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# END OF ANALYSIS
# -------------------------------------------------------
print("\n==============================")
print(" ✅ ANALYSIS COMPLETED SUCCESSFULLY ✅")
print("==============================")