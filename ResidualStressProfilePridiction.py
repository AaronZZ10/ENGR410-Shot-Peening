import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

# Load real-life data
input_data = pd.read_csv('input.csv')
residual_stress_data = pd.read_csv('output.csv')

# Merge data based on 'Test' column and shuffle the merged data
merged_data = pd.merge(input_data, residual_stress_data, on='Test', how='left')
merged_data = merged_data.sample(frac=1, random_state=0).reset_index(drop=True)

# Print the shuffled merged data
print(merged_data)

# Features and target variable
features = ['Velocity (mm/s)', 'Angle Alpha',
            'Diameter (mm)', 'Coverage (%)', 'Depth(mm)']
X = merged_data[features]
y = merged_data['S11-S33(Mpa)']

# Normalize the feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and set up the model
#tanh 20,14; 297; 31, 50; 98,96
#relu 7,8
model = MLPRegressor(hidden_layer_sizes=(
    146,118), activation='tanh', solver='lbfgs', max_iter=200, random_state=0)

# Fit the model on the entire dataset
model.fit(X_scaled, y)

# K-Fold Cross Validation
scores = cross_val_score(model, X_scaled, y, cv=5,
                         scoring=make_scorer(r2_score))
mean_r2_score = scores.mean()

print(f"Cross-Validation R2 Scores: {scores}")
print(f"Mean R2 Score: {mean_r2_score * 100}%")

# Input from user
coverage_user = float(input("Enter coverage (in percent): "))
diameter_user = float(input("Enter diameter of the shot peening (in mm): "))
angle_user = float(input("Enter angle alpha (in degree): "))
velocity_user = float(input("Enter velocity of particles (in mm/s): "))

# Generate new prediction based on user input
depths_new = np.linspace(0, 1000, 400)  # Adjusted to range from 0 to 1000 µm
X_new = pd.DataFrame({
    'Velocity (mm/s)': [velocity_user] * 400,
    'Angle Alpha': [angle_user] * 400,
    'Diameter (mm)': [diameter_user] * 400,
    'Coverage (%)': [coverage_user] * 400,
    'Depth(mm)': depths_new
})

# Scale the new input data
X_new_scaled = scaler.transform(X_new)

# Predict using the scaled input data
predictions_new = model.predict(X_new_scaled)


# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(depths_new, predictions_new, 'r-', label='Predicted Curve')
plt.xlabel('Depth (µm)')
plt.ylabel('Residual Stress (MPa)')
plt.title('Predicted Residual Stress Profile (S11-S33)')

# Display input parameters on the graph
input_params_text = (
    f"Input parameters:\n"
    f"Coverage: {coverage_user}%\n"
    f"Diameter: {diameter_user} mm\n"
    f"Angle Alpha: {angle_user}°\n"
    f"Velocity: {velocity_user} mm/s"
)
plt.annotate(input_params_text, xy=(0.75, 0.75), xycoords='axes fraction', 
             horizontalalignment='left', verticalalignment='top', 
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.legend()
plt.grid(True)
plt.show()

