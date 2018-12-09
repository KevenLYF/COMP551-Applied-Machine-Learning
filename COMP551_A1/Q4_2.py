import numpy as np
import csv

def calcW(trainSetPath):
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
    part1 = np.dot(np.transpose(inputX), inputX)            # xTx
    part2 = np.dot(np.transpose(inputX), outputY)           # xTy

    w = np.dot(np.linalg.inv(part1), part2)                 # w = (xTx)^-1 dot xTy
    return w

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

# Calculate MSEs of five different 80-20 splits and averaged MSE
wSet = []

sumMSE = 0.0

for i in range(1, 6):
    w = calcW('Datasets/CandC-train' + str(i) + '.csv')
    wSet.append(w)
    MSE = calcMSE('Datasets/CandC-test' + str(i) + '.csv', w)
    sumMSE += MSE
    print('The MSE of number ' + str(i) + " 80-20 split is " + str(MSE))

avgMSE = sumMSE/5.0
print("The MSE averaged over 5 different 80-20 splits is " + str(avgMSE))


#  Wirte average MSE and parameter table to file Assignment1_260561054_4_2 under current directory

# data = np.array(wSet)
# data = data.T
#
# # write average MSE
# with open('Assignment1_260561054_4_2', 'w') as f:
#     f.write("The MSE averaged over 5 different 80-20 splits is " + str(avgMSE) + '\n')
#     f.write('\n')
#     f.write('The parameters learnt for 5 different 80-20 splits are : \n')
#     f.write('\n')
#     for i in range(123):
#         for j in range(5):
#             f.write(str(data[0][i][j]))
#             f.write('\t')
#         f.write('\n')
