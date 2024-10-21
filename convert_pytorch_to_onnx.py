import torch
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from infer import fuse_conv_and_bn
import pnnx
import numpy as np

image_width = 768
image_height = 432

def get_tensor(np_array, half=False, device='cpu', int8=False):
    image = torch.from_numpy(np_array).permute(2, 0, 1).float()
    
    if half:
        image = image.half()
    if int8:
        image = image.to(dtype=torch.int8)
    if device  == 'cpu' :
        return image[None].cpu()
    else:
        return image[None].cuda()

def covert(half=False, device = 'cpu'):
    
    model = NeuFlow().to(device)

    checkpoint = torch.load('neuflow_mixed.pth',  map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)

    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward

    

    model.eval()
    
    if half:
        model.half()

    model.init_bhwd(1, image_height, image_width, device, amp=half) #amp True means Half Precision    

    image_1 = np.random.randint(0, 256, (image_height, image_width, 3))
    image_2 = np.random.randint(0, 256, (image_height, image_width, 3))

    x = get_tensor(image_1, half, device)
    y = get_tensor(image_2, half, device)

    print(x.shape)
    print(y.shape)

    print(x.dtype)
    print(y.dtype)

    mod = torch.jit.trace(model, (x,y))
    
    jit_name = "neuflow.pt"
    
    if half:
        jit_name = "neuflow_half.pt"

    mod.save(jit_name)
    
    onnx_name = "neuflow.onnx"
    if half:
        onnx_name = "neuflow_half.onnx"

    # You could also try exporting to the good-old onnx
    torch.onnx.export(model, (x,y), onnx_name, opset_version=16)
    
def convert_int8(device):
    
    model = NeuFlow().to(device)

    checkpoint = torch.load('neuflow_mixed.pth',  map_location=device)

    model.load_state_dict(checkpoint['model'], strict=True)
    
    model.eval()
    
    model.init_bhwd(1, image_height, image_width, device, amp=False)
    
    model.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    
    for m in model.modules():
        if type(m) is ConvBlock:
            m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)  # update conv
            m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)  # update conv
            delattr(m, "norm1")  # remove batchnorm
            delattr(m, "norm2")  # remove batchnorm
            m.forward = m.forward_fuse  # update forward
    
    model_fp32_prepared = torch.ao.quantization.prepare(model)
    
    image_1 = np.random.randint(0, 256, (image_height, image_width, 3))
    image_2 = np.random.randint(0, 256, (image_height, image_width, 3))

    x = get_tensor(image_1, device = device)
    y = get_tensor(image_2, device = device)
    
    
    model_fp32_prepared(x, y)
    
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    
    for name, param in model_int8.named_parameters():
        print(f"Layer: {name}, Data type: {param.dtype}")
    
    print(model_int8)
    
    # mod = torch.jit.trace(model_int8, (x,y))
    
    # jit_name = "neuflow_int8.pt"
    
    # mod.save(jit_name)
    
if __name__ == "__main__":
    # covert(half=False)
    # covert(half=True, device='cuda')
    convert_int8('cpu')