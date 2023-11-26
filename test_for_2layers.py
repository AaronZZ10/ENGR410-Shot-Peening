import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

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
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize variables for tracking the best model
best_r2_score = -np.inf
best_neuron_count = (0)

# Iterate over possible combinations of neuron counts
for neurons1 in range(11,21):
    for neurons2 in range(11,21):
        model = MLPRegressor(hidden_layer_sizes=(neurons1, neurons2), activation='tanh',
                             solver='lbfgs', max_iter=200, random_state=0)
        
        # K-Fold Cross Validation
        scores = cross_val_score(model, X_scaled, y, cv=5, scoring=make_scorer(r2_score))
        mean_r2_score = scores.mean()

        # Track the best model
        if mean_r2_score > best_r2_score:
            best_r2_score = mean_r2_score
            best_neuron_count = (neurons1, neurons2)

print(f"Best neuron count: {best_neuron_count}, Best R2 score: {best_r2_score}")

