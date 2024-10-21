import torch
torch.backends.quantized.engine = 'qnnpack'
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from infer import fuse_conv_and_bn
import pnnx
import numpy as np

image_width = 768
image_height = 432

def get_tensor(np_array):
    image = torch.from_numpy(np_array).permute(2, 0, 1).float()
    return image[None].cpu()

device = torch.device('cpu')

model = NeuFlow().to(device)

checkpoint = torch.load('neuflow_mixed.pth',  map_location='cpu')

model.load_state_dict(checkpoint['model'], strict=True)

for m in model.modules():
    if type(m) is ConvBlock:
        m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
        m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
        delattr(m, "norm1")  # remove batchnorm
        delattr(m, "norm2")  # remove batchnorm
        m.forward = m.forward_fuse  # update forward

  

model.eval()

model.init_bhwd(1, image_height, image_width, 'cpu', amp=False)    

image_1 = np.random.randint(0, 256, (image_height, image_width, 3))
image_2 = np.random.randint(0, 256, (image_height, image_width, 3))

x = get_tensor(image_1)
y = get_tensor(image_2)

print(x.shape)
print(y.shape)

print(x.dtype)
print(y.dtype)

flow = model(x, y)[-1][0]