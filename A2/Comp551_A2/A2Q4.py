import numpy as np
import csv
import random


# Helper function to read CSV file and returning a np array containing the data
def csvReader(filepath):
    with open(filepath) as csvFile:
        dataSet = csv.reader(csvFile, delimiter=',')
        data = []
        for row in dataSet:
            inputTemp = []
            for i in range(len(row)):
                if(row[i] != ''):                   # somehow the initial csv data has empty string, need this step to filter it
                    inputTemp.append(float(row[i]))
            data.append(inputTemp)
    data = np.array(data, dtype='float')
    return data

# Question 4 - Generate dataset by a mixture of 3 Gaussians
def generateData(m11, m12, m13, m21, m22, m23, c1, c2, c3):
    mean11 = csvReader(m11)
    mean12 = csvReader(m12)
    mean13 = csvReader(m13)
    mean21 = csvReader(m21)
    mean22 = csvReader(m22)
    mean23 = csvReader(m23)
    cov1 = csvReader(c1)
    cov2 = csvReader(c2)
    cov3 = csvReader(c3)
    class11 = np.random.multivariate_normal(mean11[0], cov1, 2000)
    class12 = np.random.multivariate_normal(mean12[0], cov2, 2000)
    class13 = np.random.multivariate_normal(mean13[0], cov3, 2000)
    class21 = np.random.multivariate_normal(mean21[0], cov1, 2000)
    class22 = np.random.multivariate_normal(mean22[0], cov2, 2000)
    class23 = np.random.multivariate_normal(mean23[0], cov3, 2000)
    class1_temp = []
    class2_temp = []

    for i in range(2000):
        choice = np.random.choice([1,2,3,], 1, p=[0.1,0.42,0.48])
        if (choice == 1):
            class1_temp.append(class11[i])
            class2_temp.append(class21[i])
        elif (choice == 2):
            class1_temp.append(class12[i])
            class2_temp.append(class22[i])
        else:
            class1_temp.append(class13[i])
            class2_temp.append(class23[i])

    classNegative = -1 * np.ones((2000, 21), dtype='float')
    classNegative[:,:-1] = class1_temp
    classPositive = np.ones((2000, 21), dtype='float')
    classPositive[:,:-1] = class2_temp

    random.shuffle(classNegative)
    random.shuffle(classPositive)

    testSet = []
    validSet = []
    trainSet = []
    testBound = int(len(classNegative) * 0.2)
    validBound = int(len(classNegative) * 0.2)

    for i in range(2000):
        if (i < testBound):
            testSet.append(classNegative[i])
            testSet.append(classPositive[i])
        elif (i >= testBound and i < testBound + validBound):
            validSet.append(classNegative[i])
            validSet.append(classPositive[i])
        else:
            trainSet.append(classNegative[i])
            trainSet.append(classPositive[i])

    random.shuffle(testSet)
    random.shuffle(trainSet)
    random.shuffle(validSet)

    with open('hwk2_datasets/DS2-test.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(testSet)):
            csvWriter.writerow(testSet[i])

    with open('hwk2_datasets/DS2-valid.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(validSet)):
            csvWriter.writerow(validSet[i])

    with open('hwk2_datasets/DS2-train.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(trainSet)):
            csvWriter.writerow(trainSet[i])

# generateData('hwk2_datasets/DS2_c1_m1.txt', 'hwk2_datasets/DS2_c1_m2.txt', 'hwk2_datasets/DS2_c1_m3.txt',
#              'hwk2_datasets/DS2_c2_m1.txt', 'hwk2_datasets/DS2_c2_m2.txt', 'hwk2_datasets/DS2_c2_m3.txt',
#              'hwk2_datasets/DS2_Cov1.txt', 'hwk2_datasets/DS2_Cov2.txt', 'hwk2_datasets/DS2_Cov3.txt')
