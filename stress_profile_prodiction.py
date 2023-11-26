import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import simpledialog

# Load real-life data
data = pd.read_csv("input.csv")
data = data.sample(frac=1, random_state=0).reset_index(drop=True)

# Print the shuffled data
print(data)

# Features and target variable
input_features = ['Velocity (mm/s)', 'Weld Radius (mm)', 'Weld Height (mm)', 'Weld Half-width (mm)',
                  'Diameter (mm)', 'Coverage (%)', 'Depth (mm)']
X = data[input_features]
output_features = ['S11 (Mpa)', "S22 (Mpa)", 'S33 (Mpa)', 'S11-S22 (Mpa)']
y = data[output_features]

# Normalize the feature data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and set up the model
# tanh 86 198;304/sgd
# relu 79
model = MLPRegressor(hidden_layer_sizes=(198), activation='tanh',
                     solver='sgd', max_iter=200, random_state=0)

# Fit the model on the entire dataset
model.fit(X_scaled, y)

# K-Fold Cross Validation
scores = cross_val_score(model, X_scaled, y, cv=5,
                         scoring=make_scorer(r2_score))
mean_r2_score = scores.mean()

print(f"Cross-Validation R2 Scores: {scores}")
print(f"Mean R2 Score: {mean_r2_score * 100}%")

# Initialize Tkinter root
root = tk.Tk()
root.withdraw()  # Hide the main Tkinter window

# Custom dialog class for multiple inputs
class MyDialog(simpledialog.Dialog):
    def __init__(self, parent, title=None):
        super().__init__(parent, title="User inputs")

    def body(self, master):
        tk.Label(master, text="Diameter (mm):").grid(row=0)
        tk.Label(master, text="Velocity (mm/s):").grid(row=1)
        tk.Label(master, text="Coverage (%):").grid(row=2)
        tk.Label(master, text="Weld Radius (mm):").grid(row=3)
        tk.Label(master, text="Weld Height (mm):").grid(row=4)
        tk.Label(master, text="Weld Half-width (mm):").grid(row=5)
        tk.Label(master, text="Stress Component (S11, S22, S33, or S11-S22):").grid(row=6)

        self.diameter = tk.Entry(master)
        self.velocity = tk.Entry(master)
        self.coverage = tk.Entry(master)
        self.weld_radius = tk.Entry(master)
        self.weld_height = tk.Entry(master)
        self.weld_halfwidth = tk.Entry(master)
        self.stress_component = tk.Entry(master)

        self.diameter.grid(row=0, column=1)
        self.velocity.grid(row=1, column=1)
        self.coverage.grid(row=2, column=1)
        self.weld_radius.grid(row=3, column=1)
        self.weld_height.grid(row=4, column=1)
        self.weld_halfwidth.grid(row=5, column=1)
        self.stress_component.grid(row=6, column=1)
        return self.diameter  # initial focus

    def apply(self):
        self.result = [self.diameter.get(), self.velocity.get(), self.coverage.get(),
                       self.weld_radius.get(), self.weld_height.get(), self.weld_halfwidth.get(),
                       self.stress_component.get()]

# Get user inputs
user_input = MyDialog(root).result

# Convert inputs to appropriate data types
diameter_user = float(user_input[0])
velocity_user = float(user_input[1])
coverage_user = float(user_input[2])
weld_radius_user = float(user_input[3])
weld_height_user = float(user_input[4])
weld_halfwidth_user = float(user_input[5])
stress_component = user_input[6]

# Generate new prediction based on user input
depths_new = np.linspace(0, 1000, 400)  # Depth range as per user input
X_new = pd.DataFrame({
    'Velocity (mm/s)': [velocity_user] * 400,
    'Weld Radius (mm)': [weld_radius_user] * 400,
    'Weld Height (mm)': [weld_height_user] * 400,
    'Weld Half-width (mm)': [weld_halfwidth_user] * 400,
    'Diameter (mm)': [diameter_user] * 400,
    'Coverage (%)': [coverage_user] * 400,
    'Depth (mm)': depths_new
})

# Scale the new input data
X_new_scaled = scaler.transform(X_new)

# Predict using the scaled input data
predictions_new = model.predict(X_new_scaled)

# Extracting the selected stress component for visualization
stress_component_index = output_features.index(stress_component + " (Mpa)")
predicted_stress = predictions_new[:, stress_component_index]

# Visualize the results
plt.figure(figsize=(10, 6))
plt.plot(depths_new, predicted_stress, 'r-',
         label=f'Predicted {stress_component}')
plt.xlabel('Depth (mm)')
plt.ylabel(f'{stress_component} (MPa)')
plt.title(f'Predicted Residual Stress Profile ({stress_component})')

# Display input parameters on the graph
input_params_text = (
    f"Input parameters:\n"
    f"Coverage: {coverage_user}%\n"
    f"Diameter: {diameter_user} mm\n"
    f"Velocity: {velocity_user} mm/s\n"
    f"Weld Radius: {weld_radius_user} mm\n"
    f"Weld Height: {weld_height_user} mm\n"
    f"Weld Half-width: {weld_halfwidth_user} mm"
)
plt.annotate(input_params_text, xy=(0.75, 0.55), xycoords='axes fraction',
             horizontalalignment='left', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

# Filter the original data to match the input parameters
filtered_data = data[(data['Velocity (mm/s)'] == velocity_user) &
                     (data['Weld Radius (mm)'] == weld_radius_user) &
                     (data['Weld Height (mm)'] == weld_height_user) &
                     (data['Weld Half-width (mm)'] == weld_halfwidth_user) &
                     (data['Diameter (mm)'] == diameter_user) &
                     (data['Coverage (%)'] == coverage_user)]

if not filtered_data.empty:
    # Extract the original stress component values
    original_stress = filtered_data[stress_component + ' (Mpa)']
    original_depths = filtered_data['Depth (mm)']

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.plot(depths_new, predicted_stress, 'r-',
            label=f'Predicted {stress_component}')
    plt.scatter(original_depths*1000, original_stress, c='blue',
                label=f'Original {stress_component}', alpha=0.7)  # Scatter plot for original data
    plt.xlabel('Depth (mm)')
    plt.ylabel(f'{stress_component} (MPa)')
    plt.title(
        f'Predicted vs. Original Residual Stress Profile ({stress_component})')

    # Display input parameters on the graph
    input_params_text = (
        f"Input parameters:\n"
        f"Coverage: {coverage_user}%\n"
        f"Diameter: {diameter_user} mm\n"
        f"Velocity: {velocity_user} mm/s\n"
        f"Weld Radius: {weld_radius_user} mm\n"
        f"Weld Height: {weld_height_user} mm\n"
        f"Weld Half-width: {weld_halfwidth_user} mm"
    )
    plt.annotate(input_params_text, xy=(0.75, 0.55), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.legend()
plt.grid(True)
plt.show()
