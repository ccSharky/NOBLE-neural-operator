import torch

# Load the training file
train_path = "nsforcing_128/nsforcing_train_128.pt"
train_data = torch.load(train_path)

# See the keys
print("Keys in training sample:", train_data.keys())

# Check the shape of each tensor
for key, value in train_data.items():
    print(f"{key}: {value.shape}")
