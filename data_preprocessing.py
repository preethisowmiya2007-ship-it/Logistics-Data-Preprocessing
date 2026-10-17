# Logistics Data Collection, Cleaning and Preprocessing
# Week 2 Task

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Sample logistics dataset
data = {
    "Shipment_ID": [101, 102, 103, 104, 105, 106, 107, 108],
    "Distance_km": [120, 250, 180, 300, np.nan, 220, 150, 1000],
    "Delivery_Time_days": [2, 4, 3, 5, 4, np.nan, 3, 15],
    "Shipping_Cost": [500, 900, 650, 1200, 800, 750, np.nan, 5000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# 1. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Handle missing values using median
df["Distance_km"] = df["Distance_km"].fillna(df["Distance_km"].median())
df["Delivery_Time_days"] = df["Delivery_Time_days"].fillna(
    df["Delivery_Time_days"].median()
)
df["Shipping_Cost"] = df["Shipping_Cost"].fillna(
    df["Shipping_Cost"].median()
)

# 3. Detect outliers using IQR method
Q1 = df[["Distance_km", "Delivery_Time_days", "Shipping_Cost"]].quantile(0.25)
Q3 = df[["Distance_km", "Delivery_Time_days", "Shipping_Cost"]].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\nOutlier Limits:")
print("Lower Limit:")
print(lower_limit)
print("Upper Limit:")
print(upper_limit)

# 4. Remove outliers
for column in ["Distance_km", "Delivery_Time_days", "Shipping_Cost"]:
    df = df[
        (df[column] >= lower_limit[column])
        & (df[column] <= upper_limit[column])
    ]

# 5. Normalize numerical data using Min-Max Scaling
scaler = MinMaxScaler()

columns_to_normalize = [
    "Distance_km",
    "Delivery_Time_days",
    "Shipping_Cost"
]

df[columns_to_normalize] = scaler.fit_transform(
    df[columns_to_normalize]
)

print("\nCleaned and Normalized Dataset:")
print(df)

# 6. Save the processed dataset
df.to_csv("processed_logistics_data.csv", index=False)

print("\nPreprocessing completed successfully.")
print("Processed dataset saved as processed_logistics_data.csv")