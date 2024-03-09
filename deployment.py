import numpy as np
import cv2
import vitis_ai_library
import xir
from IPython.display import Image

# file path
MODEL_PATH = "./outputs/my_mnist_test.xmodel"
IMG_PATH = "./my_mnist_test.jpg"

g = xir.Graph.deserialize(MODEL_PATH)
runner = vitis_ai_library.GraphRunner.create_graph_runner(g)

# input buffer
inputDim = tuple(runner.get_inputs()[0].get_tensor().dims)
inputData = [np.empty(inputDim, dtype=np.int8)]

# input image
image = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

# normalization
image = image/255.0

# quantization
fix_point = runner.get_input_tensors()[0].get_attr("fix_point")
scale = 2 ** fix_point
image = (image * scale).round()
image = image.astype(np.int8)

# set input data
inputData[0][0] = image.reshape(28, 28, 1)

# output buffer
outputData = runner.get_outputs()

# prediction
job_id = runner.execute_async(inputData, outputData)
runner.wait(job_id)

resultList = np.asarray(outputData[0])[0]
resultIdx = resultList.argmax()
resultVal = resultList[resultIdx]
print("Predictions")
for i, x in enumerate(resultList):
    print("%d : %f"%(i, x))

del runner