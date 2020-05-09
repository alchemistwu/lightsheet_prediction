import os
import numpy as np
import random
import pandas as pd
from copy import deepcopy
import cv2
from src.model import *
from tensorflow.keras.callbacks import ModelCheckpoint
def loadSplitTxt(txtPath):
    assert os.path.exists(txtPath)
    imgPaths = []
    labelPaths = []
    with open(txtPath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.strip("\n")
            imgPaths.append(item.split('#')[0])
            labelPaths.append(item.split('#')[1])
    return imgPaths, labelPaths

def data_generator(txtPath, batchSize=1):
    imgPaths, labelPaths = loadSplitTxt(txtPath)
    index = 0
    num_samples = len(imgPaths)
    while True:
        if index * batchSize > num_samples:
            index = 0
        batch_start = index * batchSize
        batch_end = (index + 1) * batchSize
        if batch_end > num_samples:
            batch_end = num_samples
        batchPathX, batchPathY = imgPaths[batch_start: batch_end], labelPaths[batch_start: batch_end]
        batchX, batchY = [], []
        for imgPath, labelPath in zip(batchPathX, batchPathY):
            batchX.append(cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE))
            print(labelPath)
            batchY.append(np.load(labelPath))
        batchX = np.asarray(batchX, dtype=np.float32)
        batchY = np.asarray(batchY, dtype=np.float32)
        yield (batchX, batchY)

def train(learning_rate=0.001, batchSize=32, epochs=100):
    weights_folder = os.path.join('weights')
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    train_txt = os.path.join("..", "dataset", "train.txt")
    val_txt = os.path.join("..", "dataset", "val.txt")
    inputs, xception_inputs, ans = get_model()
    callbackList = [ModelCheckpoint(os.path.join(weights_folder, 'model.tf'), save_best_only=True, save_weights_only=True)]
    m = Model(inputs=[inputs, xception_inputs], output=[ans])

    def categorical_crossentropy(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred, from_logits=True)

    m.compile(optimizer=keras.optimizers.RMSprop(lr=learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['accuracy']
                  )
    train_generator = data_generator(train_txt, batchSize=batchSize)
    val_generator = data_generator(val_txt, batchSize=batchSize)
    with open(train_txt, 'r') as f:
        train_steps = len(f.readlines()) // batchSize + 1
    with open(val_txt, 'r') as f:
        val_steps = len(f.readlines()) // batchSize + 1
    history = model.fit_generator(train_generator, steps_per_epoch=train_steps,
                                      epochs=epochs, validation_data=val_generator,
                                  validation_steps=val_steps,
                                      callbacks=callbackList)
    write_summary(history)

def write_summary(history):
    folder = os.path.join('result')
    if not os.path.exists(folder):
        os.mkdir(folder)

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    result_file_path = os.path.join(folder, "logs.txt")
    result_file = open(result_file_path, 'w')
    for i in range(0, len(loss)):
        result_file.write(
            '{}: train_loss={}, val_loss={}, train_acc={}, val_acc={}\n'.format(i, loss[i], val_loss[i], acc[i],
                                                                                val_acc[i]))
    result_file.close()


if __name__ == '__main__':
    generator = data_generator(os.path.join("..", "dataset", "train.txt"))
    x, y = next(generator)
    print("OK")