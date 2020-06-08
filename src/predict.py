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

def predictScan(tifPath):
    x, oriX = prepareScanForPredict(tifPath)
    weights_folder = os.path.join('..', 'weights')
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    scan_folder = os.path.join('..', 'scan')
    if not os.path.exists(scan_folder):
        os.mkdir(scan_folder)
    inputs, xception_inputs, ans = get_model()
    m = Model(inputs=[inputs, xception_inputs], outputs=[ans])
    best_model_weights = os.path.join(weights_folder,
                                      [item for item in os.listdir(weights_folder) if ".index" in item][0].replace(
                                          ".index", ""))
    m.load_weights(best_model_weights)
    print("Weights have been loaded!")
    predictions = m.predict(x, batch_size=4)

    for i in range(predictions.shape[0]):
        prediction = label2Color(predict2Mask(predictions[i]))
        # input = np.asarray(x[0][i], dtype='uint8')
        input = oriX[i]
        prediction = cv2.resize(prediction, (input.shape[1], input.shape[0]))
        cv2.imwrite(os.path.join(scan_folder, str(i) + '_predict.png'), prediction)
        cv2.imwrite(os.path.join(scan_folder, str(i) + '_input.png'), input)

def calculateVolume(tifPath,
                    widthRatio=1.4, heightRatio=1.4, thicknessRatio=5,
                    colorDict=None):
    if not colorDict:
        global COLOR_DICT
        colorDict = COLOR_DICT
    assert os.path.isdir(tifPath), 'Path not exist!'
    numDict = {}
    for img in [cv2.imread(imgPath) for imgPath in os.listdir(tifPath)]:
        imgArray = np.asarray(img)
        for key in colorDict.keys():
            if key not in numDict.keys():
                numDict[key] = 0
            numDict[key] += np.asarray(imgArray == colorDict[key], dtype=np.float).sum() * widthRatio * heightRatio * thicknessRatio

    print(numDict)



if __name__ == '__main__':
    args = parser.ArgumentParser(description='Model training arguments')

    args.add_argument('-tif', '--tifScanPath', type=str, default=None,
                      help='tif path')

    args.add_argument('-volume', '--tifVolumePath', type=str, default=None,
                      help='tif path')

    args.add_argument('-threshold', '--threshold', type=str, default=0.5,
                      help='threshold')

    parsed_arg = args.parse_args()
    if parsed_arg.tifScanPath:
        predictScan(parsed_arg.tifScanPath)
    elif parsed_arg.tifVolumePath:
        calculateVolume(parsed_arg.tifVolumePath)
    else:
        predict(threshold=float(parsed_arg.threshold))