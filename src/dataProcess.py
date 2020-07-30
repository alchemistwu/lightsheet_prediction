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

def readTif(tifPath, keepThreshold=100, imgShape=(608, 608), filterDark=True, returnOriginal=False):
    assert os.path.exists(tifPath)
    imgStack = io.imread(tifPath)
    (steps, height, width) = imgStack.shape
    print("Image Stack Size:", imgStack.shape)
    processedStacks = []
    originalStacks = []
    for step in range(steps):
        img8 = cv2.normalize(imgStack[step], None, 0, 255, cv2.NORM_MINMAX)
        img8 = np.asarray(img8, dtype='uint8')
        blur = cv2.GaussianBlur(img8, (3, 3), 0)
        imgResize = cv2.resize(blur, imgShape)
        img3Channel = cv2.cvtColor(imgResize, cv2.COLOR_GRAY2RGB)
        if filterDark:
            if np.max(imgResize) >= keepThreshold:
                processedStacks.append(img3Channel)
        else:
            processedStacks.append(img3Channel)
        if returnOriginal:
            imgOrignal = cv2.normalize(imgStack[step], None, 0, 255, cv2.NORM_MINMAX)
            imgOrignal = np.asarray(imgOrignal, dtype='uint8')
            originalStacks.append(imgOrignal)


        # cv2.imshow('im8', imgResize)
        # cv2.waitKey()
    if returnOriginal:
        return processedStacks, originalStacks
    else:
        return processedStacks

def saveTifStack(tifStack, saveFolder, startID=None):
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    assert os.path.isdir(saveFolder)
    if not startID:
        imgID = len(os.listdir(saveFolder))
    else:
        imgID = startID
    for tif in tifStack:
        cv2.imwrite(os.path.join(saveFolder, str(imgID) + '.png'), tif)
        imgID += 1

def generateImages(tifFolder=None, saveFolder=None, startID=None):
    if not tifFolder:
        tifFolder = os.path.join('..', 'dataset', 'raw')
    if not saveFolder:
        saveFolder = os.path.join('..', 'dataset', 'images')
    if not os.path.exists(saveFolder):
        os.mkdir(saveFolder)
    tifPaths = [os.path.join(tifFolder, file) for file in os.listdir(tifFolder) if file.endswith('.tif')]
    for tifPath in tifPaths:
        tifStack = readTif(tifPath)
        if startID:
            saveTifStack(tifStack, saveFolder, startID)
        saveTifStack(tifStack, saveFolder)


def generateMasks(imgFolder=os.path.join('..', 'dataset', 'images'),
                  jsonFolder=os.path.join('..', 'dataset', 'json'),
                  labelFolder=os.path.join('..', 'dataset', 'label'),
                  imgShape=(608, 608), patient=100):
    global LABEL_DICT
    numClass = len(LABEL_DICT.keys())
    labelDict = loadJson(jsonFolder)
    tmpFolder = os.path.join('..', 'dataset', 'tmp')
    if not os.path.exists(tmpFolder):
        os.mkdir(tmpFolder)
    if not os.path.exists(labelFolder):
        os.mkdir(labelFolder)

    for key in labelDict.keys():
        failFlag = False
        if not labelDict[key]:
            if os.path.exists(os.path.join(imgFolder, key)):
                os.remove(os.path.join(imgFolder, key))
        else:
            initMask = np.zeros(shape=(imgShape[0], imgShape[1], numClass), dtype='uint8')
            for labelKey in labelDict[key].keys():
                if os.path.exists(os.path.join(tmpFolder, labelKey + '.png')):
                    os.remove(os.path.join(tmpFolder, labelKey + '.png'))
                retryTimes = 0
                while not os.path.exists(os.path.join(tmpFolder, labelKey + '.png')):
                    try:
                        urllib.request.urlretrieve(labelDict[key][labelKey], os.path.join(tmpFolder, labelKey + '.png'))
                    except:
                        print(labelDict[key][labelKey])
                        print("retrying...")
                        retryTimes += 1
                        if retryTimes > patient:
                            failFlag = True
                            break
                if failFlag:
                    if os.path.exists(os.path.join(imgFolder, key)):
                        os.remove(os.path.join(imgFolder, key))
                    break
                else:
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

def label2Color(labelMask, background=True, normal=True, stroke=True):
    copyMask = copy.deepcopy(labelMask)
    canvas = np.zeros(shape=(copyMask.shape[0], copyMask.shape[1], 3), dtype='uint8')
    keys_to_include = []
    if background:
        keys_to_include.append('background')
    if normal:
        keys_to_include.append('normal')
    if stroke:
        keys_to_include.append('stroke')

    for key in keys_to_include:
        canvas[copyMask[:, :, LABEL_DICT[key]] == 1, :] = COLOR_DICT[key]
    return canvas

def loadSplitTxt(txtPath):
    assert os.path.exists(txtPath)
    imgPaths = []
    labelPaths = []
    with open(txtPath, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            item = line.strip("\n")
            imgPaths.append(item.split('#')[0])
            labelPaths.append(item.split('#')[1])
    return imgPaths, labelPaths

def data_generator(txtPath, batchSize=1, debug=False, aug=True):
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
            if aug:
                rot, flip, shiftX, shiftY = getAugmentationParameters()
            else:
                rot, flip, shiftX, shiftY = 0, 0, 0, 0
            shiftX, shiftY = 0, 0
            if aug:
                img = dataAugmentation(img, rot, flip, shiftX, shiftY, labelMask=False)
                label = dataAugmentation(label, rot, flip, shiftX, shiftY, labelMask=True)
            if debug:
                cv2.imshow('img', img)
                cv2.imshow('label', label2Color(label))
                cv2.waitKey()
            batchX.append(img)
            batchY.append(label)

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

def dataAugmentation(img, rot, flip, shiftX, shiftY, labelMask=False):
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
    if labelMask:
        shiftImg[:, :, 0] = 1
    shiftImg[mom(oy):non(oy), mom(ox):non(ox)] = processedImg[mom(-oy):non(-oy), mom(-ox):non(-ox)]

    return shiftImg

def prepareScanForPredict(tifPath):
    processedStacks, originalStacks = readTif(tifPath, imgShape=(608, 608), filterDark=False, returnOriginal=True)
    x = np.asarray(processedStacks, dtype=np.float32)
    return [x, x], originalStacks


def getAugmentationParameters():
    rot = random.randint(0, 3)
    flip = random.randint(0, 2)
    shiftX = random.randint(-200, 200)
    shiftY = random.randint(-200, 200)
    return rot, flip, shiftX, shiftY
