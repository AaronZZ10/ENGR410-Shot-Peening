import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(0)
depths = np.linspace(0, 1000, 400)


def smooth_transition(depth):
    # Generate a soft curve for the transition at 200 µm
    if depth <= 200:
        residual_stress = -400 + depth * 2.25
    else:
        # Use a polynomial to ensure a smooth transition from the end of the previous section
        transition = (depth - 200) * (depth - 200) * 0.00005
        residual_stress = 50 - transition
    return residual_stress


residual_stresses = [smooth_transition(
    depth) + np.random.normal(0, 10) for depth in depths]

data = pd.DataFrame({'depth': depths, 'residual_stress': residual_stresses})

X = data[['depth']]
y = data['residual_stress']

# ANN Model with 'tanh' activation
model = MLPRegressor(hidden_layer_sizes=(
    25), activation='tanh', solver='adam', max_iter=5000, random_state=0)
model.fit(X, y)
predictions = model.predict(X)

# Compute Mean Squared Error
mse = mean_squared_error(y, predictions)
print(f"Mean Squared Error: {mse}")

# Visualize the results using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(X['depth'].values, y, 'o', label='Actual Data')
plt.plot(X['depth'].values, predictions, 'r-',
         label='Predicted Curve using ANN with tanh')
plt.xlabel('Depth (µm)')
plt.ylabel('Residual Stress (MPa)')
plt.title('Residual Stress Profile')
plt.legend()
plt.grid(True)
plt.show()
