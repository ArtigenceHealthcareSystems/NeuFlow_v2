"""
Benchmarking different model exports

"""
import os
import onnxruntime
import torch
from NeuFlow.neuflow import NeuFlow
from NeuFlow.backbone_v7 import ConvBlock
from infer import fuse_conv_and_bn
import numpy as np
from functools import wraps
import time
import wandb
from matplotlib import pyplot as plt
from convert_pytorch_to_onnx import get_tensor

image_width = 768
image_height = 432

# Dictionary to store timing information for the current iteration
function_times = {}


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def timeit_function(func):
    """A decorator to time functions and store the time in seconds for the current iteration."""
    @wraps(func)
    def wrapper(*args, **kwargs):

        start_time = time.perf_counter()  # Start time in seconds
        result = func(*args, **kwargs)    # Call the original function
        end_time = time.perf_counter()    # End time in seconds
        
        time_taken = end_time - start_time  # Time taken in seconds
        function_name = func.__name__
        
        # Store the time taken for the current iteration in seconds
        function_times[function_name] = time_taken
        
        print(f"Function '{func.__name__}' took {time_taken:.6f} seconds in this iteration")
        return result  # Return the original function result without modification

    return wrapper

def load_pytorch_model(half=False, device='cpu'):

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

    model.init_bhwd(1, image_height, image_width, device, amp=half)   
    
    return model

def load_torch_jit(half=False, device = "cpu", int8 = False):
    if half:
        loaded_model = torch.jit.load("neuflow_half.pt", map_location=torch.device(device))
    else:
        loaded_model = torch.jit.load("neuflow.pt", map_location=torch.device(device))


    if int8:
        loaded_model = torch.jit.load("neuflow_int8.pt", map_location=torch.device(device))
        
    # Set the model to evaluation mode
    loaded_model.eval()
    
    return loaded_model

def load_onnx(half=False):
    
    if half:
        ort_session = onnxruntime.InferenceSession("neuflow_half.onnx", providers=["CPUExecutionProvider"])
    else:
        ort_session = onnxruntime.InferenceSession("neuflow.onnx", providers=["CPUExecutionProvider"])

    return ort_session

@timeit_function
def infer_onnx(ort_session,x,y):
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x), ort_session.get_inputs()[1].name: to_numpy(y)}
    ort_outs = ort_session.run(None, ort_inputs)
    
    return ort_outs

@timeit_function
def inference_torchjit(model, x, y):
    # Perform inference
    with torch.no_grad():  # No need to track gradients for inference
        flow = model(x,y)
        
    return flow
        


@timeit_function
def inference_pytorch(model, x, y):
    flow = model(x, y)[-1][0]
    
    return flow
    
    
if __name__ == "__main__":
    
    print(onnxruntime.get_available_providers())
    
    os.environ['WANDB_DISABLED'] = 'true'
    
    experiment_title = "benchmarking neuflow"
    
    wandb.init(project="benchmarking neuflow", name=experiment_title)
    
    table = wandb.Table(columns=["deployment","time"])

    half = False
    
    device = "cpu"
    
    image_1 = np.random.randint(0, 256, (image_height, image_width, 3))
    image_2 = np.random.randint(0, 256, (image_height, image_width, 3))

    x = get_tensor(image_1, half, device)
    y = get_tensor(image_2, half, device)
    
    # #Torch
    # torchModel = load_pytorch_model(half, device)
    # result = inference_pytorch(torchModel, x, y)
    # print(result[0].shape)
    # table.add_data("pytorch", function_times["inference_pytorch"])    
    # print(function_times)
    # for name, param in torchModel.named_parameters():
    #     print(f"Layer: {name}, Data type: {param.dtype}")
    #     break

    # #torch_jit
    # jitmodel = load_torch_jit(half, device)
    # result = inference_torchjit(jitmodel,x,y)
    # print(result[0].shape)
    # print(function_times)
    # table.add_data("torch_jit", function_times["inference_torchjit"])
    # for name, param in jitmodel.named_parameters():
    #     print(f"Layer: {name}, Data type: {param.dtype}")
    #     break
    # #onnx
    # onnxsession = load_onnx(half)
    # result = infer_onnx(onnxsession,x,y)
    # print(result[0].shape)
    # table.add_data("onnx", function_times["infer_onnx"])
    
    #torchjitt in int8
    jitmodel = load_torch_jit(half=False, device="cpu", int8=True)
    x = get_tensor(image_1, half=False, device="cpu")
    y = get_tensor(image_2, half=False, device="cpu")
    result = inference_torchjit(jitmodel,x,y)
    for name, param in jitmodel.named_parameters():
        print(f"Layer: {name}, Data type: {param.dtype}")
        break
    
    # Create a bar chart using matplotlib
    deployments = list(function_times.keys())
    times = list(function_times.values())

    plt.figure(figsize=(10, 5))
    bars = plt.bar(deployments, times, color='blue')
    plt.title('Function Time Benchmark')
    plt.xlabel('Deployment Type')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()

    # Add time values on top of each bar
    for bar, time in zip(bars, times):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(time, 3), ha='center', va='bottom')

    # Save the plot as an image file
    plt.savefig("bar_chart_with_labels.png")

    # Log the image to Weights and Biases
    wandb.log({"bar_chart_image": wandb.Image("bar_chart_with_labels.png")})
    
    # Log the table data
    wandb.log({"function_times_table": table})

    # Finish the wandb session
    wandb.finish()