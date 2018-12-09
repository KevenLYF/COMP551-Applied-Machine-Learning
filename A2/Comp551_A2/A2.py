import numpy as np
import csv
from sklearn import preprocessing

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

# Question 2
def gda(filePath):
    trainData = csvReader(filePath)
    sum1 = np.zeros(trainData[0].shape, dtype='float')
    sum2 = np.zeros(trainData[0].shape, dtype='float')
    count1 = 0;
    count2 = 0;
    for i in range(len(trainData)):
        if (trainData[i][len(trainData[0])-1] == -1):
            sum1 = np.add(sum1,trainData[i])
            count1 += 1
        elif (trainData[i][len(trainData[0])-1] == 1):
            sum2 = np.add(sum2, trainData[i])
            count2 += 1
    m1 = np.divide(sum1, count1)
    m2 = np.divide(sum2, count2)
    m1 = m1[:-1].reshape(1,-1) # remove labeled 1s and -1s at the last column
    m2 = m2[:-1].reshape(1,-1)
    P1 = count1/len(trainData)
    P2 = count2/len(trainData)
    sum = np.zeros((len(m1), len(m2)))
    for i in range(len(trainData)):
        part1 = np.subtract(trainData[i][:-1], m1)
        part2 = np.subtract(trainData[i][:-1], m2)
        S1 = np.dot(np.transpose(part1), part1)
        S2 = np.dot(np.transpose(part2), part2)
        sum = np.add(np.add(S1, S2), sum)

    cov = np.divide(sum, count1+count2)
    cov_inv = np.linalg.inv(cov)
    term1 = 1/2 * np.dot(np.dot(m1, cov_inv), np.transpose(m1))
    term2 = 1/2 * np.dot(np.dot(m2, cov_inv), np.transpose(m2))
    w0 = np.subtract(term2, term1) + np.log(P1) - np.log(P2)
    w1 = np.dot(cov_inv, np.transpose(np.subtract(m1, m2)))
    w = [w0, w1]
    print('w0 = ' + str(w0[0][0]))
    print('w1 = ' + str(w1))
    return w


def validateGDA(filepath, w):
    data = csvReader(filepath)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    w0 = w[0]
    w1 = w[1]
    for i in range(len(data)):
        a = w0[0][0] + np.dot(data[i][:-1], w1)
        if (a > 0):
            if(data[i][len(data[0]) - 1] == -1):
                TN += 1
            else:
                FN += 1
        elif (a < 0):
            if (data[i][len(data[0]) - 1] == 1):
                TP += 1
            else:
                FP += 1
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1measure = (2*precision*recall) / (precision+recall)
    print('Accuracy = ' + str(accuracy))
    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 measure = ' + str(F1measure))

# Question 3
# Apply Euclidean distance
def KNN(X_train, X_test, k):
    d = []
    for i in range(len(X_train)):
        sum = 0
        for j in range(len(X_test) - 1):
            sum += np.square(X_train[i][j]-X_test[j])
        d.append([np.sqrt(sum), i, X_train[i][len(X_train[0]) - 1]])

    d = sorted(d)
    nn = []
    for i in range (k):
        nn.append(d[i][2])
    prediction = np.sum(nn)
    if (prediction > 0):
        return 1
    else:
        return -1

def validateKNN(trainDataset, testDataset):
    trainData = csvReader(trainDataset)
    trainData[:, :-1] = preprocessing.normalize(trainData[:, :-1], axis=0)  # need to normalize the data
    testData = csvReader(testDataset)
    testData[:, :-1] = preprocessing.normalize(testData[:, :-1], axis=0)

    bestk = 0
    bestF1 = 0

    for i in range (1, 6):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for j in range(len(testData)):
            prediction = KNN(trainData, testData[j], i)
            if (prediction > 0):
                if (trainData[j][len(trainData[0]) - 1] == -1):
                    TN += 1
                else:
                    FN += 1
            else:
                if (trainData[j][len(trainData[0]) - 1] == 1):
                    TP += 1
                else:
                    FP += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1measure = (2 * precision * recall) / (precision + recall)
        print('k = ' + str(i))
        print(' F1 Measure = ' + str(F1measure))
        if (F1measure > bestF1):
            bestF1 = F1measure
            bestk = i

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(testData)):
        prediction = KNN(trainData, testData[i], bestk)
        if (prediction > 0):
            if (trainData[i][len(trainData[0]) - 1] == -1):
                TN += 1
            else:
                FN += 1
        else:
            if (trainData[i][len(trainData[0]) - 1] == 1):
                TP += 1
            else:
                FP += 1
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1measure = (2 * precision * recall) / (precision + recall)
    print('Best K = ' + str(bestk))
    print('Accuracy = ' + str(accuracy))
    print('Precision = ' + str(precision))
    print('Recall = ' + str(recall))
    print('F1 measure = ' + str(F1measure))


# w = gda('hwk2_datasets/DS1-train.csv')
# validateGDA('hwk2_datasets/DS1-test.csv', w)
# validateKNN('hwk2_datasets/DS1-train.csv', 'hwk2_datasets/DS1-test.csv')

# w = gda('hwk2_datasets/DS2-train.csv')
# validateGDA('hwk2_datasets/DS2-test.csv', w)
# validateKNN('hwk2_datasets/DS2-train.csv', 'hwk2_datasets/DS2-test.csv')
