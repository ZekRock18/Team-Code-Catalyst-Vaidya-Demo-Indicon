import torch
import intel_extension_for_pytorch as ipex

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")
print(f"Intel Extension for PyTorch Version: {ipex.__version__}")

# Check if MKL-DNN is enabled
print(f"MKL-DNN Enabled: {torch.backends.mkldnn.enabled}")
