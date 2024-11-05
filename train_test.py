import torch
import torch.nn.functional as F

# Example batch of label maps (3D tensor representing a batch of segmentation masks)
batch_size = 2
height = 3
width = 3
num_classes = 4

# Simulated batch of 2D label maps with shape (batch_size, height, width)
batch_label_map = torch.tensor([
    [[0, 1, 2],
     [2, 1, 0],
     [1, 0, 2]],
    
    [[1, 2, 0],
     [0, 2, 1],
     [2, 1, 0]]
], dtype=torch.int64)  # Shape: (batch_size, height, width)

# Convert the batch to one-hot encoded format
# Reshape to (batch_size * height * width) to apply one-hot and then reshape back
one_hot_encoded = F.one_hot(batch_label_map, num_classes=num_classes)  # Shape: (batch_size, height, width, num_classes)

# Rearrange to shape (batch_size, num_classes, height, width)
one_hot_encoded = one_hot_encoded.permute(0, 3, 1, 2)  # Shape: (batch_size, num_classes, height, width)

print(one_hot_encoded.shape)  # Should print (batch_size, num_classes, height, width)
print(one_hot_encoded)
