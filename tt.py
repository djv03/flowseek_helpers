
import torch
print("Torch version:", torch._version_)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA version:", torch.version.cuda)
print("Compiled with CUDA:", torch.backends.cuda.is_built())
