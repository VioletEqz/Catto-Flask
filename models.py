# The class containing the model
import onnxruntime
import numpy as np
from utils import *

class Catto:
    def __init__(self):
        self.model = onnxruntime.InferenceSession('Catto.onnx', None)
        
    def preprocess(self,input):
        cv2img = cv2.imread(input)
        img = letterbox(cv2img,scaleFill=False,auto=False)[0]
        # Conversion
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = img.astype(np.float32)
        img = np.ascontiguousarray(img)
        self.original = img
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img,0)
        return img

    def infer(self,input):
        img = self.preprocess(input)
        preds = (self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: img}))
        preds = non_max_suppression(np.array(preds))[0]
        return preds

class Face:
    def __init__(self):
        self.model = onnxruntime.InferenceSession('Catto-Face.onnx', None)
    
    def preprocess(self,input):
        cv2img = cv2.imread(input)
        img = letterbox(cv2img,scaleFill=False,auto=False)[0]
        # Conversion
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = img.astype(np.float32)
        img = np.ascontiguousarray(img)
        self.original = img
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = np.expand_dims(img,0)
        return img

    def infer(self,input):
        img = self.preprocess(input)
        preds = (self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: img}))
        preds = non_max_suppression(np.array(preds))[0]
        return preds
