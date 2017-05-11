import os,sys
import cv2
import gc
import csv
import random

def hexStringToDec(hexString):
    return int(hexString, 16)

def asciiCodeToCharacter(asciiCode):
    return chr(asciiCode)

""" 
[0..9]  <=> '0'..'9'
[10..35] <=> 'a'..'z'
[36..61] <=> 'A'..'Z'
"""

def characterToClassNumber(character):
    ret = 0
    if '0' <= character and character <= '9':
        ret = ord(character) - ord('0')
    elif 'a' <= character and character <= 'z':
        ret = 10 + ord(character) - ord('a')
    else:
        ret = 10 + 26 + ord(character) - ord('A')
    return ret

def classNumberToClassVector(classNumber):
    ret = [0] * 62
    ret[classNumber] = 1
    return ret

def hexStringToClassVector(hexString):
    return classNumberToClassVector(characterToClassNumber(asciiCodeToCharacter(hexStringToDec(hexString))))

def classVectorToCharacter(classVector):
    index = classVector.index(1.0)
    ret = ''
    if 0 <= index and index <= 9:
        ret = chr(index + ord('0'))
    elif 10 <= index and index <= 35:
        ret = chr(index - 10 + ord('a'))
    elif 36 <= index and index <= 61:
        ret = chr(index - 10 - 26 + ord('A'))
    else:
        ret = 'error: index not in [0, 61]'
    return ret

def readImagesFromFolder(baseFolderPath, typeOfData):
    dataX = []
    folderWithActualImages = os.path.join(baseFolderPath, typeOfData)
    for currentFileName in os.listdir(folderWithActualImages):
        currentImagePath = os.path.join(folderWithActualImages, currentFileName)
        img = cv2.imread(currentImagePath, 0)
        
        linearizedImg = []
        for i in range(len(img)):
            for j in range(len(img[i])):
                x = 1 if img[i, j] == 255 else 0
                linearizedImg.append(x)
        dataX.append(linearizedImg)
        
    
    unreachableObjects = gc.collect()
    print("unreachableObjects: " + str(unreachableObjects))
    
    return dataX



def getAllLabeledData(pathOfProcessedImages):
    trainDataX = []
    trainDataY = []
    testDataX = []
    testDataY = []
    
    print("started reading images from " + pathOfProcessedImages)

    for folderHex in os.listdir(pathOfProcessedImages):
        characterAsciiCode = hexStringToDec(folderHex)
        characterFolderPath = os.path.join(pathOfProcessedImages, folderHex)
        currentTrainDataX = []
        currentTrainDataY = []
        currentTestDataX = []
        currentTestDataY = []

        currentTrainDataX = readImagesFromFolder(characterFolderPath, "train")
        currentTrainDataY = hexStringToClassVector(folderHex) * len(currentTrainDataX)
        print(folderHex + ": train data reading - Done.");
        
        currentTestDataX = readImagesFromFolder(characterFolderPath, "test")
        currentTestDataY = hexStringToClassVector(folderHex) * len(currentTestDataX)
        print(folderHex + ": test data reading - Done.");

        trainDataX.extend(currentTrainDataX)
        trainDataY.extend(currentTrainDataY)
        
        testDataX.extend(currentTestDataX)
        testDataY.extend(currentTestDataY)

    trainData = [trainDataX, trainDataY]
    testData = [testDataX, testDataY]
    return [trainData, testData]

def getDataFromCSVFile(csvPath, logStatus):
    print("Started reading data from " + csvPath) if logStatus else None
    count = 0
    
    with open(csvPath, "r", newline='') as csvFile:
        rowReader = csv.reader(csvFile)
        for row in rowReader:
            print("reading line " + str(count)) if logStatus and count%10000 == 0 else None
            count += 1
            #rowWithInts = [(int(x) == 1) for x in row]
            #ret.append(rowWithInts)
    ret = [''] * count
    
    count = 0
    with open(csvPath, "r", newline='') as csvFile:
        rowReader = csv.reader(csvFile)
        for row in rowReader:
            print("reading line " + str(count)) if logStatus and count%10000 == 0 else None
            concatenatedRow = "".join(row)
            ret[count] = concatenatedRow
            count += 1
    gc.collect()
    return ret

def getAllLabeledDataFromCSVFiles(trainDataXPath, trainDataYPath, testDataXPath, testDataYPath, logStatus):
    trainDataX = getDataFromCSVFile(trainDataXPath, logStatus)
    trainDataY = getDataFromCSVFile(trainDataYPath, logStatus)
    testDataX = getDataFromCSVFile(testDataXPath, logStatus)
    testDataY = getDataFromCSVFile(testDataYPath, logStatus)
    
    trainData = [trainDataX, trainDataY]
    testData = [testDataX, testDataY]
    
    return [trainData, testData]    

def appendRowsInCSVFile(csvPath, rowsVector):
    with open(csvPath, "a", newline='') as csvFile:
        rowWriter = csv.writer(csvFile)
        for row in rowsVector:
            rowAsString = row[:]
            for j in range(len(rowAsString)):
                rowAsString[j] = str(rowAsString[j])
            rowWriter.writerow(rowAsString)

def createCSVFiles(pathOfProcessedImages, trainDataXPath, trainDataYPath, testDataXPath, testDataYPath):
    print("started reading images from " + pathOfProcessedImages)

    for folderHex in os.listdir(pathOfProcessedImages):
        characterAsciiCode = hexStringToDec(folderHex)
        characterFolderPath = os.path.join(pathOfProcessedImages, folderHex)
        currentTrainDataX = []
        currentTrainDataY = []
        currentTestDataX = []
        currentTestDataY = []

        currentTrainDataX = readImagesFromFolder(characterFolderPath, "train")
        currentTrainDataY = [hexStringToClassVector(folderHex)] * len(currentTrainDataX)
        print(folderHex + ": train data reading - Done.");
        
        currentTestDataX = readImagesFromFolder(characterFolderPath, "test")
        currentTestDataY = [hexStringToClassVector(folderHex)] * len(currentTestDataX)
        print(folderHex + ": test data reading - Done.");

        appendRowsInCSVFile(trainDataXPath, currentTrainDataX)
        appendRowsInCSVFile(trainDataYPath, currentTrainDataY)
        appendRowsInCSVFile(testDataXPath, currentTestDataX)
        appendRowsInCSVFile(testDataYPath, currentTestDataY)

def getClassesLocationIndices(data):
    # feed with data like this: data = [dataX, dataY]
    left = 0
    right = 0
    ret = []
    for i in range(len(data[1])):
        if i > 0 and data[1][i] != data[1][i-1]:
            right = i - 1
            ret.append((left, right))
            left = i
    ret.append((left, len(data[1]) - 1))
    return ret

def getBatchOfDataAsStrings(data, classesIndices, totalCount = 1000, examplesPerClass = 100):
    # feed with data like this: data = [dataX, dataY]
    # call with totalCount = -1 OR examplesPerClass = -1
    dataX = data[0]
    dataY = data[1]
    retX = []
    retY = []
    if examplesPerClass != -1:
        for (left, right) in classesIndices:
            randomPositions = random.sample(range(left, right+1), examplesPerClass)
            for j in randomPositions:
                retX.append(dataX[j])
                retY.append(dataY[j])
    elif totalCount != -1:
        baseCountOfExamplesPerClass = int(totalCount / 62)
        count = 0
        remainder = totalCount % 62
        for (left, right) in classesIndices:
            actualCountOfExamplesPerClass = baseCountOfExamplesPerClass
            if count < remainder:
                actualCountOfExamplesPerClass += 1
            count += 1
            randomPositions = random.sample(range(left, right+1), actualCountOfExamplesPerClass)
            for j in randomPositions:
                retX.append(dataX[j])
                retY.append(dataY[j])
    else:
        print("both 'totalCount' and 'examplesPerClass' are equal to -1")
    return [retX, retY]


def getBatchOfDataAsFloats(data, classesIndices, totalCount = 1000, examplesPerClass = 100):
    # feed with data like this: data = [dataX, dataY]
    # call with totalCount = -1 OR examplesPerClass = -1
    dataX = data[0]
    dataY = data[1]
    retX = []
    retY = []
    if examplesPerClass != -1:
        for (left, right) in classesIndices:
            randomPositions = random.sample(range(left, right+1), examplesPerClass)
            for j in randomPositions:
                floatDataX = [float(x) for x in dataX[j]]
                floatDataY = [float(y) for y in dataY[j]]
                retX.append(floatDataX)
                retY.append(floatDataY)
    elif totalCount != -1:
        baseCountOfExamplesPerClass = int(totalCount / 62)
        count = 0
        remainder = totalCount % 62
        for (left, right) in classesIndices:
            actualCountOfExamplesPerClass = baseCountOfExamplesPerClass
            if count < remainder:
                actualCountOfExamplesPerClass += 1
            count += 1
            randomPositions = random.sample(range(left, right+1), actualCountOfExamplesPerClass)
            for j in randomPositions:
                floatDataX = [float(x) for x in dataX[j]]
                floatDataY = [float(y) for y in dataY[j]]
                retX.append(floatDataX)
                retY.append(floatDataY)
    else:
        print("both 'totalCount' and 'examplesPerClass' are equal to -1")
    return [retX, retY]

def testBatchOfDataVisually(batchOfData):
    for i in range(0, len(batchOfData[0]), 10):
        dataX = batchOfData[0][i]
        graphicLetter = ""
        for j in range(0,32):
            graphicLetter = graphicLetter + "".join([('0' if x == 0.0 else ' ') for x in dataX[j * 32 : (j+1)*32]]) + "\n"
        dataY = batchOfData[1][i]
        character = classVectorToCharacter(dataY)
        print(graphicLetter)
        print(character)
        _ = input()

            