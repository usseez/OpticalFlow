import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from models.PWCNet import PWCDCNet


model = PWCDCNet()

module = model.conv1a[0]
# print(list(module.named_parameters()))
print(list(module.named_buffers()))

parameters_to_prune = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        parameters_to_prune.append((m, "weight"))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.RandomUnstructured,
    amount=0.3,
)

# print(list(module.named_parameters()))

print(list(module.named_buffers()))