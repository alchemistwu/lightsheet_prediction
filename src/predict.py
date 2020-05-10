import os
import numpy as np
import random
import pandas as pd
from copy import deepcopy
import cv2
from model import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow
from dataProcess import *
import argparse as parser
try:
    from src.model import *
except:
    pass
try:
    from src.dataProcess import *
except:
    pass

def predict(threshold=0.5, batchSize=4):
    weights_folder = os.path.join('weights')
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)

    val_txt = os.path.join("..", "dataset", "val.txt")
    inputs, xception_inputs, ans = get_model()
    m = Model(inputs=[inputs, xception_inputs], outputs=[ans])
    best_model_weights = os.path.join(weights_folder,
                                      [item for item in os.listdir(weights_folder) if ".index" in item][0].replace(
                                          ".index", ""))
    m.load_weights(best_model_weights)
    print("Weights have been loaded!")
    val_generator = data_generator(val_txt, batchSize=batchSize)
    with open(val_txt, 'r') as f:
        val_steps = len(f.readlines()) // batchSize + 1
    result = m.predict_generator(val_generator)
    print(result.shape)

    for i in range(result.shape[0]):
        binarayMask = np.asarray(result[i] > threshold, dtype='uint8')
        img = label2Color(binarayMask)
        cv2.imshow('test', img)
        cv2.waitKey()

if __name__ == '__main__':
    args = parser.ArgumentParser(description='Model training arguments')

    args.add_argument('-threshold', '--threshold', type=str, default=0.5,
                      help='threshold')

    parsed_arg = args.parse_args()

    predict(threshold=float(parsed_arg.threshold))