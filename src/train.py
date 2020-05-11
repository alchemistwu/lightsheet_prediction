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




def train(learning_rate=0.001, batchSize=32, epochs=100):
    weights_folder = os.path.join('weights')
    if not os.path.exists(weights_folder):
        os.mkdir(weights_folder)
    train_txt = os.path.join("..", "dataset", "train.txt")
    val_txt = os.path.join("..", "dataset", "val.txt")
    inputs, xception_inputs, ans = get_model()
    callbackList = [ModelCheckpoint(os.path.join(weights_folder, 'model.tf'), save_best_only=True, save_weights_only=True)]
    m = Model(inputs=[inputs, xception_inputs], outputs=[ans])

    def categorical_crossentropy(y_true, y_pred):
        return tensorflow.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

    m.compile(optimizer=tensorflow.keras.optimizers.RMSprop(lr=learning_rate),
                  loss=categorical_crossentropy,
                  metrics=['accuracy']
                  )
    train_generator = data_generator(train_txt, batchSize=batchSize)
    val_generator = data_generator(val_txt, batchSize=batchSize)
    with open(train_txt, 'r') as f:
        train_steps = len(f.readlines()) // batchSize + 1
    with open(val_txt, 'r') as f:
        val_steps = len(f.readlines()) // batchSize + 1
    history = m.fit_generator(train_generator, steps_per_epoch=train_steps,
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
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    result_file_path = os.path.join(folder, "logs.txt")
    result_file = open(result_file_path, 'w')
    for i in range(0, len(loss)):
        result_file.write(
            '{}: train_loss={}, val_loss={}, train_acc={}, val_acc={}\n'.format(i, loss[i], val_loss[i], acc[i],
                                                                                val_acc[i]))
    result_file.close()


if __name__ == '__main__':
    args = parser.ArgumentParser(description='Model training arguments')

    args.add_argument('-eph', '--epochs', type=int, default=100,
                      help='Number of epochs')

    args.add_argument('-batches', '--batch_size', type=int, default=8,
                      help='Number of batches per train')

    parsed_arg = args.parse_args()


    train(batchSize=parsed_arg.batch_size,
          epochs=parsed_arg.epochs)