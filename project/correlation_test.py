import numpy as np
import torch
from torch.autograd import Variable

from project.models.Correlation.Correlation_Module.spatial_correlation_sampler.spatial_correlation_sampler \
    import spatial_correlation_sample

input0 = np.random.normal(10, 1, (1, 1, 200, 176))
input1 = np.random.normal(10, 1, (1, 1, 200, 176))
input0 = Variable(torch.from_numpy(input0)).float()
input1 = Variable(torch.from_numpy(input1)).float()

output = spatial_correlation_sample(input0, input1, kernel_size=1, patch_size=5, stride=1, padding=0, dilation_patch=2)
print(output)
print(output.shape)