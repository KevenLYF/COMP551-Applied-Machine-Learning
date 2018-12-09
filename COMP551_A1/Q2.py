import numpy as np
import math
import csv
import sys
import matplotlib.pyplot as plt

# 20-degree polynomial
M = 20

def calcW(trainSetPath):
    with open(trainSetPath) as csvFile:
        trainDataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []
        inputX_temp = []
        outputY_temp = []
        x0 = []   #x0 is just an array with 1s
        for row in trainDataSet:
            inputX_temp.append(float(row[0]))
            outputY_temp.append(float(row[1]))

        for i in range(len(inputX_temp)):  #add x0 to matrix
            x0.append(1.0)

    inputX.append(x0)

    for j in range(1, M+1, 1):
        inputX.append(np.power(inputX_temp, j))

    outputY.append(outputY_temp)
    inputX = np.transpose(np.array(inputX, dtype = 'float'))
    outputY = np.transpose(np.array(outputY, dtype = 'float'))
    part1 = np.dot(np.transpose(inputX), inputX)            # xTx
    part2 = np.dot(np.transpose(inputX), outputY)           # xTy

    w = np.dot(np.linalg.inv(part1), part2)                 # w = (xTx)^-1 dot xTy
    return w



def calcMSE(dataSetPath, w):
    with open(dataSetPath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []
        inputX_temp = []
        outputY_temp = []
        x0 = []   # x0 is just an array with 1
        for row in dataSet:
            inputX_temp.append(float(row[0]))
            outputY_temp.append(float(row[1]))

        for i in range(len(inputX_temp)):  # add x0 to matrix
            x0.append(1.0)

    inputX.append(x0)

    for j in range(1, M+1, 1):
        inputX.append(np.power(inputX_temp, j))

    outputY.append(outputY_temp)
    inputX = np.transpose(np.array(inputX, dtype = 'float'))
    outputY = np.transpose(np.array(outputY, dtype = 'float'))

    diff = np.subtract(np.dot(inputX, w), outputY)  # y(x, w) - y(x)
    MSE = np.square(diff).mean()
    return MSE


wMatrix = calcW("Datasets/Dataset_1_train.csv")
print("MSE for Training set (No regularization) = " + str(calcMSE("Datasets/Dataset_1_train.csv", wMatrix)))
print("MSE for Validation set (No regularization) = " + str(calcMSE("Datasets/Dataset_1_valid.csv", wMatrix)))

# ---------------------------------------  Plot function  ------------------------------------

def plotModel(dataSetPath, wMatrix):
    with open(dataSetPath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []
        modelDataPoint = []

        for row in dataSet:
            inputX.append(float(row[0]))
            outputY.append(float(row[1]))

        for i in range(len(inputX)):
            modelDataPoint.append(calcModelOutput(inputX[i], wMatrix))

    plt.plot(sorted(inputX), sorted(modelDataPoint), 'b-', label='Model')
    plt.plot(inputX, outputY, 'ro', label="Data")
    plt.legend()
    plt.title("Fit Visualization")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()



def calcModelOutput(x, wMatrix):
    y = 0.0
    for i in range(0, M+1, 1):
        y += math.pow(x, i) * wMatrix[i][0]
    return y

#plotModel("Datasets/Dataset_1_valid.csv", wMatrix)

# ---------------------------------------  L2 Regularization part  ------------------------------------

# Equaion w = (x^Tx + λI)^(-1) dot (x^Ty)

def calcW_L2(dataSetPath, ld):
    with open(dataSetPath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []
        inputX_temp = []
        outputY_temp = []
        x0 = []   # x0 is just an array with 1
        for row in dataSet:
            inputX_temp.append(float(row[0]))
            outputY_temp.append(float(row[1]))

        for i in range(len(inputX_temp)):  # add x0 to matrix
            x0.append(1.0)

    inputX.append(x0)

    for j in range(1, M+1, 1):
        inputX.append(np.power(inputX_temp, j))

    outputY.append(outputY_temp)
    inputX = np.transpose(np.array(inputX, dtype = 'float'))
    outputY = np.transpose(np.array(outputY, dtype = 'float'))

    part1 = np.dot(np.transpose(inputX), inputX)        # x^Tx
    part2 = ld * np.identity(part1.shape[0])            # λI
    left = np.linalg.inv(np.add(part1, part2))          # (x^Tx + λI)^(-1)
    right = np.dot(np.transpose(inputX), outputY)       # x^Ty

    wMatrix_L2 = np.dot(left, right)

    return wMatrix_L2

def pickLambda(trainDataSetPath, validDataSetPath):
    ld = 0.0
    ld_set = []
    train_MSE = []
    valid_MSE = []

    while (ld <= 1):
        wMatrix_L2 = calcW_L2(trainDataSetPath, ld)
        train_MSE.append(calcMSE(trainDataSetPath, wMatrix_L2))
        valid_MSE.append(calcMSE(validDataSetPath, wMatrix_L2))
        ld_set.append(ld)
        ld += 0.005

    printMinLambda(ld_set, valid_MSE)
    # plot training MSE and validation MSE
    # plt.plot(ld_set, train_MSE, 'r-', label = "Train MSE")
    # plt.plot(ld_set, valid_MSE, 'b-', label = "Valid MSE")
    # plt.legend()
    # plt.axis([0, 1, 5, 15])
    # plt.title("MSE VS Lambda")
    # plt.xlabel("Lambda")
    # plt.ylabel("MSE")
    # plt.show()

def printMinLambda(ld_set, valid_MSE):
    minLambda = 1
    minMSE = sys.float_info.max
    for i in range(len(valid_MSE)):
        minMSE = min(minMSE, valid_MSE[i])
        minLambda = ld_set[valid_MSE.index(minMSE)]
    print("Minimum MSE: " + str(minMSE) + "\n" + "Lambda : " + str(minLambda))

pickLambda("Datasets/Dataset_1_train.csv", "Datasets/Dataset_1_valid.csv")

# Calculate the test performance by using this λ
wMatrix_L2 = calcW_L2("Datasets/Dataset_1_train.csv", 0.02)
print("MSE of Test data set (L2 regularization) : " + str(calcMSE("Datasets/Dataset_1_test.csv", wMatrix_L2)))

# plotModel("Datasets/Dataset_1_valid.csv", wMatrix_L2)
# plotModel("Datasets/Dataset_1_test.csv", wMatrix_L2)