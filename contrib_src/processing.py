import numpy as np
import os
import mxnet as mx
from modelhublib.processor import ImageProcessorBase
import PIL
import SimpleITK
import numpy as np
import json

class ImageProcessor(ImageProcessorBase):

    def _preprocessBeforeConversionToNumpy(self, image):
        if isinstance(image, PIL.Image.Image):
            # resize here
            image = image.resize((224,224), resample = PIL.Image.LANCZOS)
            image = np.array(image).astype(np.float32)
            if len(image.shape) > 2:
                image = image[:,:,0:3]
            else:
                image = np.stack((image,)*3, axis=-1)
            return image
        else:
            raise IOError("Image Type not supported for preprocessing.")

    def _preprocessAfterConversionToNumpy(self, npArr):
        # reshape, convert to batch, and float32
        arr = mx.nd.array(npArr.reshape(1, 3,224,224).astype(np.float32))
        return arr

    def computeOutput(self, inferenceResults):
        probs = np.squeeze(np.asarray(inferenceResults))
        with open("model/labels.json") as jsonFile:
            labels = json.load(jsonFile)
        result = []
        for i in range (len(probs)):
            obj = {'label': str(labels[str(i)]),
                    'probability': float(probs[i])}
            result.append(obj)
        return result
