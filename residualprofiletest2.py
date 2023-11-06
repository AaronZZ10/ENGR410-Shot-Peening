import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score

# Load real-life data
input_data = pd.read_csv('input.csv')
residual_stress_data = pd.read_csv('output.csv')

# Merge data based on 'Test' column
merged_data = pd.merge(input_data, residual_stress_data, on='Test', how='left').reset_index(drop=True)

# Shuffle the merged data
merged_data = merged_data.sample(frac=1, random_state=0)

# Print the shuffled merged data
print(merged_data)

# Features and target variable
features = ['Velocity (mm/s)', 'Angle Alpha', 'Diameter (mm)', 'Coverage (%)', 'Depth(mm)']
X = merged_data[features]
y = merged_data['S11-S33(Mpa)']


# Define the model
model = MLPRegressor(hidden_layer_sizes=(1,9), activation='relu', solver='lbfgs', max_iter=50000, random_state=0)

# Fit the model and perform cross-validation
model.fit(X, y)
scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(r2_score))

print(f"Cross-Validation R2 Scores: {scores}")
print(f"Mean R2 Score: {scores.mean()*100}%")

# Input from user
coverage_user = float(input("Enter coverage (in percent): "))
diameter_user = float(input("Enter diameter of the shot peening (in mm): "))
angle_user = float(input("Enter angle alpha (in degree): "))
velocity_user = float(input("Enter velocity of particles (in mm/s): "))

# Generate new prediction based on user input
depths_new = np.linspace(0, 1, 400) * 1000  # Adjusted to range from 0 to 1000 µm
X_new = pd.DataFrame({
    'Velocity (mm/s)': [velocity_user] * 400,
    'Angle Alpha': [angle_user] * 400,
    'Diameter (mm)': [diameter_user] * 400,
    'Coverage (%)': [coverage_user] * 400,
    'Depth(mm)': depths_new
})

predictions_new = model.predict(X_new)

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(depths_new, predictions_new, 'r-', label='Predicted Curve')
plt.xlabel('Depth (µm)')
plt.ylabel('Residual Stress (MPa)')
plt.title('Predicted Residual Stress Profile based on User Input')
plt.ylim([-400, 200])  # Ensure the y-axis range matches your expected stress range
plt.legend()
plt.grid(True)
plt.show()
