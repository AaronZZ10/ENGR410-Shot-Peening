import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score

# Load real-life data
input_data = pd.read_csv('input.csv')
residual_stress_data = pd.read_csv('output.csv')

# Merge data based on 'Test' column
merged_data = pd.merge(input_data, residual_stress_data, on='Test', how='left')

# Print the merged data
print(merged_data)

# Features and target variable
features = ['Velocity (mm/s)', 'Angle Alpha',
            'Diameter (mm)', 'Coverage (%)', 'Depth(mm)']
X = merged_data[features]
y = merged_data['S11-S33(Mpa)']

model = MLPRegressor(hidden_layer_sizes=(25,), activation='tanh',
                     solver='lbfgs', max_iter=5000, random_state=0)

# K-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5, scoring=make_scorer(r2_score))

print(f"Cross-Validation R2 Scores: {scores}")
print(f"Mean R2 Score: {scores.mean()}")

# Splitting data for further visualization
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

model.fit(X_train, y_train)

# Input from user
coverage_user = float(input("Enter coverage (in percent): "))
diameter_user = float(input("Enter diameter of the shot peening (in mm): "))
angle_user = float(input("Enter angle alpha (in degree): "))
velocity_user = float(input("Enter velocity of particles (in mm/s): "))

# Generate new prediction based on user input
# Adjust this based on the real-life depth range
depths_new = np.linspace(0, 1, 400)
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
plt.xlabel('Depth (mm)')
plt.ylabel('Residual Stress S11-S33 (MPa)')
plt.title('Predicted Residual Stress Profile based on User Input')
plt.legend()
plt.grid(True)
plt.show()
