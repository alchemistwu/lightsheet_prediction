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
    weights_folder = os.path.join('..', 'weights')
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    result_folder = os.path.join('..', 'result')
    if not os.path.exists(result_folder):
        os.mkdir(result_folder)
    val_txt = os.path.join("..", "dataset", "val.txt")
    inputs, xception_inputs, ans = get_model()
    m = Model(inputs=[inputs, xception_inputs], outputs=[ans])
    best_model_weights = os.path.join(weights_folder,
                                      [item for item in os.listdir(weights_folder) if ".index" in item][0].replace(
                                          ".index", ""))
    m.load_weights(best_model_weights)
    print("Weights have been loaded!")
    val_generator = data_generator(val_txt, batchSize=batchSize, aug=False)
    with open(val_txt, 'r') as f:
        val_steps = len(f.readlines()) // batchSize + 1
    index = 0
    for i in range(val_steps):
        x, y = next(val_generator)
        predictions = m.predict_on_batch(x)
        for j in range(predictions.shape[0]):
            prediction = label2Color(predict2Mask(predictions[j]))
            input = np.asarray(x[0][j], dtype='uint8')
            label = label2Color(np.asarray(y[j], dtype='uint8'))
            cv2.imwrite(os.path.join(result_folder, str(index) + '_predict.png'), prediction)
            cv2.imwrite(os.path.join(result_folder, str(index) + '_input.png'), input)
            cv2.imwrite(os.path.join(result_folder, str(index) + '_label.png'), label)
            # cv2.imshow('predict', prediction)
            # cv2.imshow('input', input)
            # cv2.imshow('label', label)
            # cv2.waitKey()
            index += 1


if __name__ == '__main__':
    args = parser.ArgumentParser(description='Model training arguments')

    args.add_argument('-threshold', '--threshold', type=str, default=0.5,
                      help='threshold')

    parsed_arg = args.parse_args()

    predict(threshold=float(parsed_arg.threshold))