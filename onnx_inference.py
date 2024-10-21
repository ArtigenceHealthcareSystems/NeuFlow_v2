import onnxruntime
from convert_pytorch_to_onnx import get_tensor
import numpy as np


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

image_width = 768
image_height = 432

ort_session = onnxruntime.InferenceSession("neuflow.onnx", providers=["CPUExecutionProvider"])

image_1 = np.random.randint(0, 256, (image_height, image_width, 3))
image_2 = np.random.randint(0, 256, (image_height, image_width, 3))

image_1 = image_1.astype(np.float32)
image_2 = image_2.astype(np.float32)

x = get_tensor(image_1)
y = get_tensor(image_2)

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x), ort_session.get_inputs()[1].name: to_numpy(y)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_outs[0].shape)