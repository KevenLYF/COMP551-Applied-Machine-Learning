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


# Question 1 - Generate dataset
def generateDataset(covPath, m0Path, m1Path):
    cov = csvReader(covPath)
    mean0 = csvReader(m0Path)
    mean1 = csvReader(m1Path)
    c0 = np.random.multivariate_normal(mean0[0], cov, 2000)
    classNegative = -1 * np.ones((2000, 21), dtype='float')
    classNegative[:,:-1] = c0
    c1 = np.random.multivariate_normal(mean1[0], cov, 2000)
    classPositive = np.ones((2000, 21), dtype='float')
    classPositive[:,:-1] = c1

    random.shuffle(classNegative)
    random.shuffle(classPositive)
    testBound = int(len(classNegative) * 0.2)
    validBound = int(len(classNegative) * 0.2)

    testSet = []
    validSet = []
    trainSet = []
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

    with open('hwk2_datasets/DS1-test.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(testSet)):
            csvWriter.writerow(testSet[i])

    with open('hwk2_datasets/DS1-valid.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(validSet)):
            csvWriter.writerow(validSet[i])

    with open('hwk2_datasets/DS1-train.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(trainSet)):
            csvWriter.writerow(trainSet[i])

# generateDataset('hwk2_datasets/DS1_Cov.txt','hwk2_datasets/DS1_m_0.txt', 'hwk2_datasets/DS1_m_1.txt')