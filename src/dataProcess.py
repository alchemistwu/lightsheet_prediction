from skimage import io
import numpy as np
import os
import cv2
import urllib
import json
import copy
import random

LABEL_DICT = {'background': 0, 'normal': 1, 'stroke': 2}
COLOR_DICT = {'background': (0, 0, 0), 'normal': (0, 255, 0), 'stroke': (255, 0, 0)}

def readTif(tifPath, keepThreshold=100, imgShape=(608, 608)):
    assert os.path.exists(tifPath)
    imgStack = io.imread(tifPath)
    (steps, height, width) = imgStack.shape
    print(imgStack.shape)
    processedStacks = []
    for step in range(steps):
        img8 = cv2.normalize(imgStack[step], None, 0, 255, cv2.NORM_MINMAX)
        img8 = np.asarray(img8, dtype='uint8')
        blur = cv2.GaussianBlur(img8, (3, 3), 0)
        imgResize = cv2.resize(blur, imgShape)
        img3Channel = cv2.cvtColor(imgResize, cv2.COLOR_GRAY2RGB)
        if np.max(imgResize) >= keepThreshold:
            processedStacks.append(img3Channel)

        # cv2.imshow('im8', imgResize)
        # cv2.waitKey()

    return processedStacks

def saveTifStack(tifStack, saveFolder):
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    assert os.path.isdir(saveFolder)
    imgID = len(os.listdir(saveFolder))
    for tif in tifStack:
        cv2.imwrite(os.path.join(saveFolder, str(imgID) + '.png'), tif)
        imgID += 1

def generateImages():
    tifFolder = os.path.join('..', 'dataset', 'raw')
    saveFolder = os.path.join('..', 'dataset', 'images')
    tifPaths = [os.path.join(tifFolder, file) for file in os.listdir(tifFolder) if file.endswith('.tif')]
    for tifPath in tifPaths:
        tifStack = readTif(tifPath)
        saveTifStack(tifStack, saveFolder)

def generateMasks(imgShape=(608, 608)):
    global LABEL_DICT
    numClass = len(LABEL_DICT.keys())
    jsonFolder = os.path.join('..', 'dataset', 'json')
    labelDict = loadJson(jsonFolder)
    imgFolder = os.path.join('..', 'dataset', 'images')
    labelFolder = os.path.join('..', 'dataset', 'label')
    tmpFolder = os.path.join('..', 'dataset', 'tmp')
    if not os.path.exists(tmpFolder):
        os.mkdir(tmpFolder)
    if not os.path.exists(labelFolder):
        os.mkdir(labelFolder)
    for key in labelDict.keys():
        if not labelDict[key]:
            if os.path.exists(os.path.join(imgFolder, key)):
                os.remove(os.path.join(imgFolder, key))
        else:
            initMask = np.zeros(shape=(imgShape[0], imgShape[1], numClass), dtype='uint8')
            for labelKey in labelDict[key].keys():
                if os.path.exists(os.path.join(tmpFolder, labelKey + '.png')):
                    os.remove(os.path.join(tmpFolder, labelKey + '.png'))
                while not os.path.exists(os.path.join(tmpFolder, labelKey + '.png')):
                    try:
                        urllib.request.urlretrieve(labelDict[key][labelKey], os.path.join(tmpFolder, labelKey + '.png'))
                    except:
                        print("retrying...")
                labelImage = cv2.imread(os.path.join(tmpFolder, labelKey + '.png'), cv2.IMREAD_GRAYSCALE)
                initMask[:, :, LABEL_DICT[labelKey]][np.equal(labelImage, 255)] = 1
            MaskLeak = copy.deepcopy(initMask)
            MaskLeak = MaskLeak.sum(axis=2) == 0
            initMask[:, :, LABEL_DICT['normal']][MaskLeak] = 1
            np.save(os.path.join(labelFolder, key.replace('.png', '.npy')), initMask)

def loadJson(jsonFolder):
    jsonPath = os.path.join(jsonFolder, os.listdir(jsonFolder)[0])
    extractedItems = {}
    with open(jsonPath) as json_file:
        data = json.load(json_file)
        for item in data:
            if len(item['Label'].keys()) == 0:
                labels = None
            else:
                labels = {}
                for label in item['Label']['objects']:
                    labels[label['value']] = label['instanceURI']
            extractedItems[item['External ID']] = labels
    return extractedItems

def label2Color(labelMask):
    copyMask = copy.deepcopy(labelMask)
    canvas = np.zeros(shape=(copyMask.shape[0], copyMask.shape[1], 3), dtype='uint8')
    print(copyMask.shape)
    for key in LABEL_DICT.keys():
        canvas[copyMask[:, :, LABEL_DICT[key]] == 1, :] = COLOR_DICT[key]
    return canvas

def genTxtTrainingSplit(split=0.2):
    imgFolder = os.path.join('..', 'dataset', 'images')
    labelFolder = os.path.join('..', 'dataset', 'label')
    items = list(os.listdir(labelFolder))
    random.shuffle(items)

    valNum = int(len(items) * split)

    fVal = open(os.path.join('..', 'dataset', 'val.txt'), 'w')
    fTrain = open(os.path.join('..', 'dataset', 'train.txt'), 'w')

    for index in range(len(items)):
        assert os.path.exists(os.path.join(imgFolder, items[index].replace('.npy', '.png'))), print(os.path.join(imgFolder, items[index].replace('.npy', 'png')))
        if index < valNum:
            fVal.write(os.path.join(imgFolder, items[index].replace('.npy', '.png')) + "#" + os.path.join(labelFolder, items[index]) + "\n")
        else:
            fTrain.write(os.path.join(imgFolder, items[index].replace('.npy', '.png')) + "#" + os.path.join(labelFolder, items[index]) + "\n")
    fVal.close()
    fTrain.close()

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

def data_generator(txtPath, batchSize=1, debug=False):
    imgPaths, labelPaths = loadSplitTxt(txtPath)
    index = 0
    num_samples = len(imgPaths)
    while True:
        if index * batchSize >= num_samples:
            index = 0
        batch_start = index * batchSize
        batch_end = (index + 1) * batchSize
        if batch_end > num_samples:
            batch_end = num_samples
        batchPathX, batchPathY = imgPaths[batch_start: batch_end], labelPaths[batch_start: batch_end]
        batchX, batchY = [], []
        for imgPath, labelPath in zip(batchPathX, batchPathY):

            img = cv2.imread(imgPath)
            label = np.load(labelPath)
            rot, flip, shiftX, shiftY = getAugmentationParameters()
            img = dataAugmentation(img, rot, flip, shiftX, shiftY)
            label = dataAugmentation(label, rot, flip, shiftX, shiftY)
            if debug:
                cv2.imshow('img', img)
                cv2.imshow('label', label2Color(label))
                cv2.waitKey()
            batchX.append(img)
            batchY.append(label)

            # batchX.append(cv2.imread(imgPath))
            # batchY.append(np.load(labelPath))
        batchX = np.asarray(batchX, dtype=np.float32)
        batchY = np.asarray(batchY, dtype=np.float32)
        index += 1
        yield ([batchX, batchX], batchY)

def predict2Mask(prediction):
    copyMask = np.zeros(shape=(prediction.shape[0], prediction.shape[1], 3), dtype='uint8')
    binarayMask = np.argmax(prediction, axis=-1)
    for key in LABEL_DICT.keys():
        label = np.zeros(shape=(len(LABEL_DICT.keys()), ), dtype='uint8')
        label[LABEL_DICT[key]] = 1
        copyMask[binarayMask == LABEL_DICT[key], :] = label
    return copyMask

def dataAugmentation(img, rot, flip, shiftX, shiftY):
    processedImg = img

    if rot == 0:
        processedImg = processedImg
    else:
        processedImg = np.rot90(processedImg, rot)

    if flip == 0:
        processedImg = processedImg
    elif flip == 1:
        processedImg = np.flipud(processedImg)
    elif flip == 2:
        processedImg = np.fliplr(processedImg)

    non = lambda s: s if s < 0 else None
    mom = lambda s: max(0, s)

    ox, oy = shiftX, shiftY

    shiftImg = np.zeros_like(processedImg)
    shiftImg[mom(oy):non(oy), mom(ox):non(ox)] = processedImg[mom(-oy):non(-oy), mom(-ox):non(-ox)]

    return shiftImg

def getAugmentationParameters():
    rot = random.randint(0, 3)
    flip = random.randint(0, 2)
    shiftX = random.randint(-200, 200)
    shiftY = random.randint(-200, 200)
    return rot, flip, shiftX, shiftY

if __name__ == '__main__':

    train_txt = os.path.join("..", "dataset", "train.txt")
    train_generator = data_generator(train_txt, batchSize=16, debug=True)
    with open(train_txt, 'r') as f:
        train_steps = len(f.readlines()) // 16 + 1
    for i in range(train_steps):
        next(train_generator)