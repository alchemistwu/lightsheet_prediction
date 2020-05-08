from skimage import io
import numpy as np
import os
import cv2
import urllib
import json
import copy

LABEL_DICT = {'background': 0, 'normal': 1, 'stroke': 2}

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

if __name__ == '__main__':
    generateMasks()