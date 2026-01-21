import torch
import sys

# Get filename from command line argument, or use default
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    print("Usage: python3 inspect_pt.py <path_to_file.pt>")
    print("\nExample: python3 inspect_pt.py nsforcing_128/nsforcing_train_128.pt")
    print("Or: python3 inspect_pt.py my_data.pt")
    sys.exit(1)

print(f"Loading: {path}")
data = torch.load(path, map_location="cpu")

print("\nKeys:", list(data.keys()))
print("\nShapes and dtypes:")
for k, v in data.items():
    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
