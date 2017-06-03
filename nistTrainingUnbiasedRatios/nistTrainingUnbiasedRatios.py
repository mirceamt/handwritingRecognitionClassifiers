import dataTools

processedImagesPath = r"F:\Processed Images\32x32UnbiasedRatio"
testDataXPath = r"F:\Processed Images\32x32UnbiasedRatio\csvs\testDataX.csv"
testDataYPath = r"F:\Processed Images\32x32UnbiasedRatio\csvs\testDataY.csv"
trainDataXPath = r"F:\Processed Images\32x32UnbiasedRatio\csvs\trainDataX.csv"
trainDataYPath = r"F:\Processed Images\32x32UnbiasedRatio\csvs\trainDataY.csv"

dataTools.createCSVFiles(processedImagesPath, trainDataXPath, trainDataYPath, testDataXPath, testDataYPath)





#import trainCNN

#pathToStoreCheckpoints = r'F:\saved checkpoints\cnn2'
#trainCNN.trainWithContinuousBatches(pathToStoreCheckpoints)

#pathToRestoreCheckpoints = r'F:\saved checkpoints\cnn2'
#pathToRestoreCheckpoints = r'F:\saved checkpoints\cnn1'
#checkpointName = "cnn1st"
#checkpointStep = "99000"
#trainCNN.testNetVisually(pathToRestoreCheckpoints, checkpointName, checkpointStep)
