import pandas as pd
import os

# Ensure 'data' directory exists
data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

# Define the Iris dataset
data = {
    "sepal_length": [5.1, 4.9, 4.7, 4.6, 5.0] * 30,
    "sepal_width": [3.5, 3.0, 3.2, 3.1, 3.6] * 30,
    "petal_length": [1.4, 1.4, 1.3, 1.5, 1.4] * 30,
    "petal_width": [0.2, 0.2, 0.2, 0.2, 0.2] * 30,
    "species": ["setosa"] * 50 + ["versicolor"] * 50 + ["virginica"] * 50
}

# Create DataFrame and save as CSV
df = pd.DataFrame(data)
csv_path = os.path.join(data_folder, "iris.csv")
df.to_csv(csv_path, index=False)

print(f"✅ Clean CSV file created at {csv_path}!")