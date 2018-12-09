import numpy as np
import csv
import matplotlib.pyplot as plt

stepSize = 1e-6

# Read training data setfrom CSV file
with open("Datasets/Dataset_2_train.csv") as csvFile:
    trainDataSet = csv.reader(csvFile, delimiter=',')
    inputX = []
    outputY = []
    xTemp = []
    yTemp = []
    x0 = []

    for row in trainDataSet:
        xTemp.append(float(row[0]))
        outputY.append(float(row[1]))
    for i in range(len(xTemp)):
        x0.append(1.0)

    inputX.append(xTemp)
    inputX.append(x0)
    inputX = np.transpose(np.array(inputX, dtype="float"))
    outputY = np.transpose(np.array(outputY, dtype = 'float'))

# Read validation data set from CSV file
with open("Datasets/Dataset_2_valid.csv") as csvFile:
    validDataSet = csv.reader(csvFile, delimiter=',')
    inputX2 = []
    outputY2 = []
    xTemp = []
    x0 = []

    for row in validDataSet:
        xTemp.append(float(row[0]))
        outputY2.append(float(row[1]))
    for i in range(len(xTemp)):
        x0.append(1.0)

    inputX2.append(xTemp)
    inputX2.append(x0)
    inputX2 = np.transpose(np.array(inputX2, dtype="float"))
    outputY2 = np.transpose(np.array(outputY2, dtype="float"))

# Read test data set from CSV file
with open("Datasets/Dataset_2_test.csv") as csvFile:
    validDataSet = csv.reader(csvFile, delimiter=',')
    inputX3 = []
    outputY3 = []
    xTemp = []
    x0 = []

    for row in validDataSet:
        xTemp.append(float(row[0]))
        outputY3.append(float(row[1]))
    for i in range(len(xTemp)):
        x0.append(1.0)

    inputX3.append(xTemp)
    inputX3.append(x0)
    inputX3 = np.transpose(np.array(inputX3, dtype="float"))
    outputY3 = np.transpose(np.array(outputY3, dtype="float"))


trainMSE_set = []
validMSE_set = []
numEpoch = 10000

def calcMSE(x, y, w):
    diff = np.subtract(np.dot(x,w), y)
    MSE = np.square(diff).mean()
    return MSE

def calcW(epoch):
    w = np.array([1.0, 1.0], dtype="float")
    for i in range(epoch):
        for j in range(inputX.shape[0]):
            w[0] = w[0] - stepSize * ((w[1] + w[0] * inputX[j][0]) - outputY[j])
            w[1] = w[1] - stepSize * ((w[1] + w[0] * inputX[j][0]) - outputY[j]) * inputX[j][0]

        trainMSE_set.append(calcMSE(inputX, outputY, w))
        validMSE_set.append(calcMSE(inputX2, outputY2, w))

    return w

wMatrix  = calcW(numEpoch)

# print("w0 is : " + str(wMatrix[0]) + "\n" + "w1 is : " + str(wMatrix[1]) + "\nwith " + str(numEpoch) + " epoch")
# print(calcMSE(inputX3, outputY3, wMatrix))

def plotLearnCurve(trainMSE_set, validMSE_set, numEpoch):
    plt.plot(trainMSE_set, 'r', label = 'Train MSE')
    plt.plot(validMSE_set, 'b', label = 'Valid MSE')
    plt.legend()
    plt.axis([0, numEpoch, 0, 30])
    plt.title("Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()

# plotLearnCurve(trainMSE_set, validMSE_set, numEpoch)


# Plot learning process, by setting epoch number from small value to large value

def plotFitEvolve(testW):
    with open("Datasets/Dataset_2_train.csv") as csvFile:
        trainDataSet = csv.reader(csvFile, delimiter=',')
        x = []
        y = []
        model_output = []

        for row in trainDataSet:
            x.append(float(row[0]))
            y.append(float(row[1]))

        for i in range(len(inputX)):
            model_output.append(testW[1] + x[i] * testW[0])


    plt.plot(x, model_output,'b',label="Train Data")
    plt.plot(x, y, 'ro', label="Train Data")
    plt.legend()
    plt.title("Learning Process")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

# Comment out to see the learning process
# testW = np.array([1.0, 1.0], dtype="float")
# for i in range (1, 6):
#     epoch = i * 1200
#     print("epoch number : " + str(epoch))
#     testW = calcW(epoch)
#     plotFitEvolve(testW)
