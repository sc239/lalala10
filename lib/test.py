import torch
'''
# Create two 1D tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# Use torch.meshgrid to generate coordinate grids
grid_x, grid_y = torch.meshgrid(x, y)

# Display the generated grids
print("Grid X:")
print(grid_x)

print("\nGrid Y:")
print(grid_y)

mesh_grid = torch.stack([grid_x, grid_y], dim=1)
print(mesh_grid)
'''
import torch.nn as nn

criterion = nn.BCELoss()

# Example usage
predicted_probabilities = torch.tensor([0.8,0.1])
true_labels = torch.tensor([0.9,0.1])

loss = criterion(predicted_probabilities, true_labels)
print(loss.item())