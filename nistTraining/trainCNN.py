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
allLabeledData = dataTools.getAllLabeledDataFromCSVFiles(trainDataXPath, trainDataYPath, testDataXPath, testDataYPath, logStatus) #correct
#allLabeledData = dataTools.getAllLabeledDataFromCSVFiles(testDataXPath, testDataYPath, testDataXPath, testDataYPath, logStatus)
print("Finished reading labeled data")

print("Started creating classes location indices in the train labeled data..")
trainClassesIndices = dataTools.getClassesLocationIndices(allLabeledData[0])
print("Finished creating classes location indices in the train labeled data..")

print("Started creating classes location indices in the test labeled data..")
testClassesIndices = dataTools.getClassesLocationIndices(allLabeledData[1])
print("Finished creating classes location indices in the test labeled data..")

# create test batch:
#batchOfData = dataTools.getRandomBatchOfDataAsFloats(allLabeledData[0], trainClassesIndices, totalCount = -1, examplesPerClass = 20)

# visually test a batch
#dataTools.testBatchOfDataVisually(batchOfData)

x = tf.placeholder(tf.float32, shape=[None, 32*32])
y_ = tf.placeholder(tf.float32, shape=[None, 62])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 40])
b_conv1 = bias_variable([40])

x_image = tf.reshape(x, [-1,32,32,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 40, 80])
b_conv2 = bias_variable([80])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 80, 2048])
b_fc1 = bias_variable([2048])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 80])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([2048, 1024])
b_fc2 = bias_variable([1024])

h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

keep_prob = tf.placeholder(tf.float32)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([1024, 62])
b_fc3 = bias_variable([62])

y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def trainWithRandomBatches():
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print("started training...")
    stepsCount = 100000
    for i in range(stepsCount):
        batchOfData = dataTools.getRandomBatchOfDataAsFloats(allLabeledData[0], trainClassesIndices, totalCount = -1, examplesPerClass = 15)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        if i%1000 == 0:
            print("creating checkpoint for step %d..." % (i))
            if i == 0:
                saver.save(sess, r"D:\Saved Models\cnn1st", global_step=i, write_meta_graph = True)
            else:
                saver.save(sess, r"D:\Saved Models\cnn1st", global_step=i, write_meta_graph = False)
            print("created checkpoint for step %d." % (i))
        train_step.run(feed_dict={x: batchOfData[0], y_: batchOfData[1], keep_prob: 0.5})

    saver.save(sess, "cnn1st", global_step=stepsCount-1, write_meta_graph = False)
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def trainWithContinuousBatches(pathForStoringCheckPoints):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    one = 0.5
    zero = -0.0
    # IMPORTANT: call this function before the creation of continuous batches
    dataTools.initClassesCurrentIndicesMap(allLabeledData[0], trainClassesIndices)
    print("started training...")
    stepsCount = 50000
    for i in range(stepsCount):
        batchOfData = dataTools.getContinuousBatchOfDataAsFloats(allLabeledData[0], trainClassesIndices, totalCount = -1, examplesPerClass = 15, one=one, zero=zero)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        if i%1000 == 0:
            print("creating checkpoint for step %d..." % (i))
            saver.save(sess, pathForStoringCheckPoints + "\\cnn", global_step=i, write_meta_graph = True)
            print("created checkpoint for step %d." % (i))
        sess.run(train_step, feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 0.5})

    saver.save(sess, pathForStoringCheckPoints + "\\cnn", global_step=stepsCount-1, write_meta_graph = True)
    #print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

def createMetaFile():
    saver = tf.train.Saver()
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver.save(sess, r"D:\Saved Models\cnn1st", global_step=1, write_meta_graph = True)

def test(pathToRestoreCheckpoints, checkpointName, checkpointStep):
    typeOfData = 1 # 0 for train, 1 for test
    # IMPORTANT: call this function before the creation of continuous batches
    dataTools.initClassesCurrentIndicesMap(allLabeledData[typeOfData], testClassesIndices)
    sessRestored = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sessRestored, pathToRestoreCheckpoints + '\\' + checkpointName + '-' + checkpointStep)
    # saver.restore(sessRestored, r'D:\Saved Models\cnn1st-99000')
    #newSaver = tf.train.import_meta_graph(r'D:\Saved Models\cnn1st-99000.meta')
    #newSaver.restore(sessRestored,  r'D:\Saved Models\cnn1st-99000')
    
    stepsCount = 10000
    for i in range(stepsCount):
        batchOfData = dataTools.getRandomBatchOfDataAsFloats(allLabeledData[typeOfData], testClassesIndices, totalCount = -1, examplesPerClass = 15)
        train_accuracy = sessRestored.run(accuracy, feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 1.0})
        #train_accuracy = accuracy.eval(feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

def testNetVisually(pathToRestoreCheckpoints, checkpointName, checkpointStep):
    typeOfData = 1 # 0 for train, 1 for test
    # IMPORTANT: call this function before the creation of continuous batches
    dataTools.initClassesCurrentIndicesMap(allLabeledData[typeOfData], testClassesIndices)
    sessRestored = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sessRestored, pathToRestoreCheckpoints + '\\' + checkpointName + '-' + checkpointStep)
    # saver.restore(sessRestored, r'D:\Saved Models\cnn1st-99000')
    #newSaver = tf.train.import_meta_graph(r'D:\Saved Models\cnn1st-99000.meta')
    #newSaver.restore(sessRestored,  r'D:\Saved Models\cnn1st-99000')
    
    stepsCount = 10000
    for i in range(stepsCount):
        batchOfData = dataTools.getRandomBatchOfDataAsFloats(allLabeledData[typeOfData], testClassesIndices, totalCount = -1, examplesPerClass = 15)
        #train_accuracy = sessRestored.run(accuracy, feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 1.0})
        #train_accuracy = accuracy.eval(feed_dict={x:batchOfData[0], y_: batchOfData[1], keep_prob: 1.0})
        #print("step %d, training accuracy %g"%(i, train_accuracy))
        for j in range(0, len(batchOfData[0]), 1):
            yPredictedVector = sessRestored.run(y_conv, feed_dict={x: [batchOfData[0][j]], keep_prob: 1.0})
            yLabeledVector = batchOfData[1][j]

            graphicLetter = ""
            for k in range(0,32):
                graphicLetter = graphicLetter + "".join([('0' if X == 0.0 else ' ') for X in batchOfData[0][j][k * 32 : (k+1) * 32]]) + "\n"
            yPredictedCharactersMultiplePossibilities = dataTools.classVectorToMultipleCharacters(yPredictedVector[0].tolist(), 5)
            # yPredictedCharacter = dataTools.classVectorToCharacter(yPredictedVector[0].tolist())
            yLabeledCharacter = dataTools.classVectorToCharacter(yLabeledVector)
            
            print(graphicLetter)
            #print("Predicted character: " + yPredictedCharacter)
            print("Possible predicted characters: " + yPredictedCharactersMultiplePossibilities)
            print("Labeled character: " + yLabeledCharacter)
            _ = input()