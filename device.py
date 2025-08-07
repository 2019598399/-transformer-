import torch
if torch.npu.is_available():
    device = "npu"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"