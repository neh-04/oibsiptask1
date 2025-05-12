import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data/iris.csv")

# Prepare features and target
X = df.drop(columns=["species"])  # Features
y = df["species"]  # Target

# Convert species labels to numeric
le = LabelEncoder()
y = le.fit_transform(y)

# ✅ Step 1: Apply Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Step 2: Apply PCA for Feature Reduction (Improves Learning)
pca = PCA(n_components=3)  # Reduce to 3 main features
X_pca = pca.fit_transform(X_scaled)

# ✅ Step 3: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# ✅ Step 4: Use Deep Learning Model (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, solver='adam', random_state=42)
model.fit(X_train, y_train)

# ✅ Step 5: Apply Cross-Validation for Better Accuracy
cross_val_scores = cross_val_score(model, X_pca, y, cv=5)
cv_accuracy = cross_val_scores.mean()

# ✅ Step 6: Save trained model
joblib.dump(model, "models/iris_model.pkl")

# ✅ Step 7: Evaluate model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Final Model trained successfully with accuracy: {accuracy * 100:.2f}%")
print(f"✅ Cross-Validation Accuracy: {cv_accuracy * 100:.2f}%")
print("✅ Model saved as models/iris_model.pkl!")