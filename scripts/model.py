import joblib

# Load trained model
model = joblib.load("models/iris_model.pkl")

# Function to predict species for given flower measurements
def predict_species(features):
    prediction = model.predict([features])
    return prediction[0]

# Example Test Case
example_features = [5.7, 2.8, 4.1, 1.3]  # Sample input
predicted_species = predict_species(example_features)

print(f"🔍 Predicted Species: {predicted_species}")