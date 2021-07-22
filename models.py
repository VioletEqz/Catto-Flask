# The class containing the model
import onnxruntime,onnx
import numpy as np
from matplotlib import pyplot as plt
from utils import *

class Catto:
    def __init__(self):
        self.model = onnxruntime.InferenceSession('Catto.onnx', None)
    
    def preprocess(self,input):
        cv2img = cv2.imread(input)
        img = letterbox(cv2img,scaleFill=False,auto=False)[0]
        # Conversion
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        self.original = img
        img = torch.from_numpy(img).to('cpu')
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        img = img.detach().cpu().numpy()
        return img

    def infer(self,input):
        img = self.preprocess(input)
        preds = torch.tensor(self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: img}))
        preds = non_max_suppression(preds)[0].detach().cpu().numpy()
        return preds



