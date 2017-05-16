import trainCNN

#pathToStoreCheckpoints = r'F:\saved checkpoints\cnn2'
#trainCNN.trainWithContinuousBatches(pathToStoreCheckpoints)

#pathToRestoreCheckpoints = r'F:\saved checkpoints\cnn2'
pathToRestoreCheckpoints = r'F:\saved checkpoints\cnn1'
checkpointName = "cnn1st"
checkpointStep = "99000"
trainCNN.testNetVisually(pathToRestoreCheckpoints, checkpointName, checkpointStep)
