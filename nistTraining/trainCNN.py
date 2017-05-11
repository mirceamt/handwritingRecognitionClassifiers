import tensorflow as tf
import dataTools
import cv2
import csv

"""
#pathOfProcessedImages = r"F:\Processed Images\32x32\\"

#img = cv2.imread(r"D:\Processed Images\32x32\6b\train\236.png", 0);
        
#linearizedImg = ""
#for i in range(len(img)):
#    for j in range(len(img[i])):
#        linearizedImg += " " if img[i][j] == 255 else "0"
#    linearizedImg += "\n"
#print(linearizedImg)


#with open(r"D:\Processed Images\test.csv", "r", newline='') as csvFile:
#    rowreader = csv.reader(csvFile)
#    for row in rowreader:
#        print(row)

#with open(r"D:\Processed Images\test1.csv", "a", newline='') as csvFile:
#    rowWriter = csv.writer(csvFile)
#    rowWriter.writerow(['0'] * 6)
#    rowWriter.writerow(['1'] * 6)
#    rowWriter.writerow(['0'] * 4 + ['1'] * 2)
    
#_ = input()
"""

pathOfProcessedImages = r"D:\Processed Images\32x32\\"
#dataTools.getAllLabeledData(pathOfProcessedImages)
trainDataXPath = r"D:\Processed Images\trainDataX.csv"
trainDataYPath = r"D:\Processed Images\trainDataY.csv"

testDataXPath = r"D:\Processed Images\testDataX.csv"
testDataYPath = r"D:\Processed Images\testDataY.csv"

#dataTools.createCSVFiles(pathOfProcessedImages, trainDataXPath, trainDataYPath, testDataXPath, testDataYPath)

logStatus = True
print("Started readin labeled data")
allLabeledData = dataTools.getAllLabeledDataFromCSVFiles(trainDataXPath, trainDataYPath, testDataXPath, testDataYPath, logStatus)
print("Finished reading labeled data")

print("Started creating classes location indices in the labeled data..")
trainClassesIndices = dataTools.getClassesLocationIndices(allLabeledData[0])
print("Finished creating classes location indices in the labeled data..")

# create test batch:
#batchOfData = dataTools.getBatchOfDataAsFloats(allLabeledData[0], trainClassesIndices, totalCount = -1, examplesPerClass = 20)

# visually test a batch
#dataTools.testBatchOfDataVisually(batchOfData)


