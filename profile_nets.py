import numpy as np
import warnings
warnings.filterwarnings("ignore")

import time

import onnx
import caffe2.python.onnx.backend
from tqdm import tqdm

# curr = []
# # # CNN
# img = np.random.randn(1, 2, 257, 61).astype(np.float32)
# model = onnx.load('mudv5_cnn_4_c_16.onnx')

# LSTM
# img = np.random.randn(1, 257, 1, 2).astype(np.float32)
# model = onnx.load('mudv5_lstm.onnx')

# FF
img = np.random.randn(1, 1, 257).astype(np.float32)
model = onnx.load('mudv5_ff.onnx')

# @profile
def forward():
    ret = caffe2.python.onnx.backend.run_model(model, [img])
    return ret

############ CNN
curr = []

# img = np.random.randn(1, 2, 257, 61).astype(np.float32)
# model = onnx.load('mudv5_cnn_6_c_16.onnx')


for i in tqdm(range(20)):
    st = time.time()
    mask = forward()
    curr.append(time.time() - st)

print(mask[0].shape)
print("DNN :: Took {} +/- {} s to process".format(np.mean(curr), np.std(curr)))

########## LSTM
# curr = []
# img = np.random.randn(1, 257, 1, 2).astype(np.float32)
# model = onnx.load('mudv5_lstm.onnx')
#
# for i in tqdm(range(20)):
#     st = time.time()
#     mask = forward()
#     curr.append(time.time() - st)
#
# print(mask[0].shape)
# print("LSTM :: Took {} +/- {} s to process".format(np.mean(curr), np.std(curr)))