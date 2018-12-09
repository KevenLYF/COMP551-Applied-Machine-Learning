import numpy as np
import csv
import sys
import math
import matplotlib.pyplot as plt

def calcW_Ridge(trainSetPath, ld):
    with open(trainSetPath) as csvFile:
        trainDataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []

        for row in trainDataSet:
            inputX_temp = []
            inputX_temp.append(1.0)
            for i in range(5, len(row)-1):
                inputX_temp.append(float(row[i]))
            inputX.append(inputX_temp)
            outputY_temp = []
            outputY_temp.append(float(row[len(row)-1]))
            outputY.append(outputY_temp)

    inputX = np.array(inputX, dtype = 'float')
    outputY = np.array(outputY, dtype = 'float')

    part1 = np.dot(np.transpose(inputX), inputX)        # x^Tx
    part2 = ld * np.identity(part1.shape[0])            # λI
    left = np.linalg.inv(np.add(part1, part2))          # (x^Tx + λI)^(-1)
    right = np.dot(np.transpose(inputX), outputY)       # x^Ty

    wMatrix_Ridge = np.dot(left, right)

    return wMatrix_Ridge

def calcMSE(dataSetPath, w):
    with open(dataSetPath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []

        for row in dataSet:
            inputX_temp = []
            inputX_temp.append(1.0)
            for i in range(5, len(row)-1):
                inputX_temp.append(float(row[i]))
            inputX.append(inputX_temp)
            outputY_temp = []
            outputY_temp.append(float(row[len(row)-1]))
            outputY.append(outputY_temp)

    inputX = np.array(inputX, dtype = 'float')
    outputY = np.array(outputY, dtype = 'float')
    diff = np.subtract(np.dot(inputX, w), outputY)  # y(x, w) - y(x)
    MSE = np.square(diff).mean()
    return MSE

def pickLambda():
    ld = 1e-10
    ld_set = []
    MSE_avg = []

    while (ld <= 1e6):
        MSE_set = []
        for i in range(1, 6):
            w = calcW_Ridge('Datasets/CandC-train' + str(i) + '.csv', ld)
            MSE_set.append(calcMSE('Datasets/CandC-test' + str(i) + '.csv', w))
        MSE_avg.append(np.array(MSE_set, dtype='float').mean())
        ld_set.append(math.log10(ld))
        ld *= 10

    printMinLambda(ld_set, MSE_avg)
    # plt.plot(ld_set, MSE_avg)
    # plt.title("MSE VS λ")
    # plt.xlabel("ln(λ)")
    # plt.ylabel("Average MSE")
    # plt.show()

def printMinLambda(ld_set, MSE_avg):
    minLambda = 0.0
    minMSE = sys.float_info.max
    for i in range(len(MSE_avg)):
        minMSE = min(minMSE, MSE_avg[i])
        minLambda = 10**ld_set[MSE_avg.index(minMSE)]
    print("Minimum average MSE: " + str(minMSE) + "\n" + "λ : " + str(minLambda))

pickLambda()

#  Wirte average MSE with best λ and parameter table to file Assignment1_260561054_4_2 under current directory

# wSet = []
# for i in range(1, 6):
#     w = calcW_Ridge('Datasets/CandC-train' + str(i) + '.csv', 1.0)
#     wSet.append(w)
#
# data = np.array(wSet)
# data = data.T

# write average MSE
# with open('Assignment1_260561054_4_3.txt', 'w') as f:
#     f.write("The MSE averaged over 5 different 80-20 splits, based on best λ = 1.0  is " + str("0.018555206975731275") + '\n')
#     f.write('\n')
#     f.write('The parameters learnt for 5 different 80-20 splits are : \n')
#     f.write('\n')
#     for i in range(123):
#         for j in range(5):
#             f.write(str(data[0][i][j]))
#             f.write('\t')
#         f.write('\n')

# ------------------------------------- Feature Selection ----------------------------------

wSet = []
for i in range(1, 6):
    w = calcW_Ridge('Datasets/CandC-train' + str(i) + '.csv', 1.0)
    wSet.append(w)

data = np.array(wSet)
data = data.T

w_avg = []
for i in range(123):
    sum = 0.0
    avg = 0.0
    for j in range(5):
        sum += data[0][i][j]
    avg = sum/5
    w_avg.append(avg)

# plt.plot(sorted(w_avg), 'ro')
# plt.title("Averaged Weight of 123 features")
# plt.xlabel("features")
# plt.ylabel("Average Weight")
# plt.show()

w_filter = []
for i in range(len(w_avg)):
    if (abs(w_avg[i]) < 0.02):
        w_filter.append(w_avg[i])

# print("There are : " + str(len(w_filter)) + " weights whose value is smaller than 0.02")

# Calculate the learning parameters w again by cutting off the irrelevant features.
index_irre = []
for i in range(len(w_filter)):
    index_irre.append(w_avg.index(w_filter[i]))

def calcW_Ridge_FS(trainSetPath, ld, irre_features):
    with open(trainSetPath) as csvFile:
        trainDataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []

        for row in trainDataSet:
            inputX_temp = []
            inputX_temp.append(1.0)
            for i in range(5, len(row)-1):
                if (i in irre_features):
                    pass
                else:
                    inputX_temp.append(float(row[i]))
            inputX.append(inputX_temp)
            outputY_temp = []
            outputY_temp.append(float(row[len(row)-1]))
            outputY.append(outputY_temp)

    inputX = np.array(inputX, dtype = 'float')
    outputY = np.array(outputY, dtype = 'float')

    part1 = np.dot(np.transpose(inputX), inputX)        # x^Tx
    part2 = ld * np.identity(part1.shape[0])            # λI
    left = np.linalg.inv(np.add(part1, part2))          # (x^Tx + λI)^(-1)
    right = np.dot(np.transpose(inputX), outputY)       # x^Ty

    wMatrix_Ridge = np.dot(left, right)

    return wMatrix_Ridge

def calcMSE_FS(dataSetPath, w, irre_features):
    with open(dataSetPath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter = ',')
        inputX = []
        outputY = []

        for row in dataSet:
            inputX_temp = []
            inputX_temp.append(1.0)
            for i in range(5, len(row)-1):
                if i in irre_features:
                    pass
                else:
                    inputX_temp.append(float(row[i]))
            inputX.append(inputX_temp)
            outputY_temp = []
            outputY_temp.append(float(row[len(row)-1]))
            outputY.append(outputY_temp)

    inputX = np.array(inputX, dtype = 'float')
    outputY = np.array(outputY, dtype = 'float')
    diff = np.subtract(np.dot(inputX, w), outputY)  # y(x, w) - y(x)
    MSE = np.square(diff).mean()
    return MSE

# Calculate MSEs and averaged MSE for 5 different 80-20 splits with reduced features
sumMSE = 0.0

for i in range(1, 6):
    w = calcW_Ridge_FS('Datasets/CandC-train' + str(i) + '.csv', 1.0, index_irre)
    MSE = calcMSE_FS('Datasets/CandC-test' + str(i) + '.csv', w, index_irre)
    sumMSE += MSE
    print('The MSE of number ' + str(i) + " 80-20 split after Feature Selection is " + str(MSE))

avgMSE = sumMSE/5.0
print("The MSE averaged over 5 different 80-20 splits after Feature Selection is " + str(avgMSE))

