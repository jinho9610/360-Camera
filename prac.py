import torch
import torch.nn as nn

m = nn.Softmax(dim=2)
input = torch.randn(2, 3)
output = m(input)

print('input := == == == == == == == == == == == == == == ===')
print(input)
print('output := == == == == == == == == == == == == == == ===')
print(output)
