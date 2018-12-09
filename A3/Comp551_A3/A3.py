from pathlib import Path
import string
import collections
import csv
import numpy as np
import matplotlib as plt
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import warnings; warnings.simplefilter('ignore')

TOP = 10000

#   -----------------------------------------   Question 1  --------------------------------------------
def preprocessing(dataset):
    translator=str.maketrans(string.punctuation, ' '*len(string.punctuation))
    filepath = Path(dataset)
    text = filepath.read_text()
    text = text.translate(translator).lower()
    return text

def genVocabFile(source, target):
    dataset = preprocessing(source)
    freqwords = collections.Counter()
    for reviews in dataset.splitlines():
        words = reviews.split()[:-1]
        freqwords.update(words)
    freqwords = freqwords.most_common(TOP)
    with open(target, "w", newline='') as textfile:
        writer = csv.writer(textfile, delimiter='\t', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(freqwords)):
            writer.writerow([freqwords[i][0], i, freqwords[i][1]])

def genVocab(source):
    dataset = preprocessing(source)
    freqwords = collections.Counter()
    for reviews in dataset.splitlines():
        words = reviews.split()[:-1]
        freqwords.update(words)
    freqwords = freqwords.most_common(TOP)
    vocab = {}
    for i in range(len(freqwords)):
        vocab[freqwords[i][0]] = (i, freqwords[i][1])
    return vocab

def buildIDFile(source, vocab, target):
    dataset = preprocessing(source)
    reviewByID = []
    count = 0
    for reviews in dataset.splitlines():
        if (len(reviews) == 0):
            continue
        words = reviews.split()
        if not (words[-1].isdigit()):
            continue
        score = int(words[len(words) - 1])
        row = []
        for i in range(len(words) - 1):
            id = vocab.get(words[i], -1)
            if (id != -1):
                row.append(id[0])
        if (len(row) > 0):
            row.append(score)
            reviewByID.append(row)
    with open(target, "w", newline='') as textfile:
        writer = csv.writer(textfile, delimiter=' ')
        for i in range(len(reviewByID)):
            row = []
            score = reviewByID[i][-1]
            for j in range(len(reviewByID[i]) - 1):
                row.append(reviewByID[i][j])
            row.append('\t' + str(score))
            writer.writerow(row)

def getReviews(datapath):
    review = []
    scores = []
    filepath = Path(datapath)
    text = filepath.read_text()
    for comments in text.splitlines():
        row = []
        IDs = comments.split()[:-1]
        score = comments.split()[-1]
        for i in range(len(IDs)):
            row.append(int(IDs[i]))
        review.append(row)
        scores.append(score)

    return (review, scores)

def buildBinaryBOW(reviewsID):
    fVector = []
    for i in range(len(reviewsID)):
        row = [0] * TOP
        uniqueSet = set()
        for j in range(len(reviewsID[i])):
            pos = reviewsID[i][j]
            if pos in uniqueSet:
                continue
            row[pos] = 1
            uniqueSet.add(pos)
        fVector.append(row)
    return fVector

def buildFreqBOW(reviewsID):
    fVector = []
    for i in range(len(reviewsID)):
        uniqueSet = set()
        row = [0] * TOP
        for j in range(len(reviewsID[i])):
            pos = reviewsID[i][j]
            if pos in uniqueSet:
                continue
            count = reviewsID[i].count(pos)
            row[pos] = count/len(reviewsID[i])
            uniqueSet.add(pos)
        fVector.append(row)
    return fVector

#   -----------------------------------------   Question 2  --------------------------------------------
# trainReviews = getReviews("output_datasets/yelp-train.txt")
# trainX = buildFreqBOW(trainReviews[0])
# trainY = trainReviews[1]
#
# validReviews = getReviews("output_datasets/yelp-valid.txt")
# validX = buildBinaryBOW(validReviews[0])
# validY = validReviews[1]
#
# testReviews = getReviews("output_datasets/yelp-test.txt")
# testX = buildBinaryBOW(testReviews[0])
# testY = testReviews[1]


def randomClf(X_train, y_train, X_test, y_test):
    clf_uniform = DummyClassifier(strategy='uniform')
    clf_uniform.fit(X_train, y_train)
    predict_uniform = []
    for i in range(len(X_test)):
        res_uniform = clf_uniform.predict([X_test[i]])
        predict_uniform.append(res_uniform)
    print('Uniform Random F1 Measure = ' + str(metrics.f1_score(y_test, predict_uniform, average='micro')))

def majorityClf(X_train, y_train, X_test, y_test):
    clf_mostFreq = DummyClassifier(strategy='most_frequent')
    clf_mostFreq.fit(X_train, y_train)
    predict_mostFreq = []
    for i in range(len(X_test)):
        res_mostFreq = clf_mostFreq.predict([X_test[i]])
        predict_mostFreq.append(res_mostFreq)
    print('Majority Class F1 Measure = ' + str(metrics.f1_score(y_test, predict_mostFreq, average='micro')))

def plotTrainningProcess(dataset):
    a = np.array(dataset,dtype='float')
    a = np.transpose(a)
    plt.plot(a[1], a[0], 'ro')
    plt.title('Finding Hyper Parameter')
    plt.xlabel('Hyper Parameter')
    plt.ylabel('F1 Measure')
    plt.show()

def BernoulliNativeBayes(X_train, y_train, X_test, y_test, a):
    clf = BernoulliNB(alpha=a)
    clf.fit(X_train, y_train)
    predict = []
    for i in range(len(X_test)):
        res = clf.predict([X_test[i]])
        predict.append(res)
    f1 = metrics.f1_score(y_test, predict, average='micro')
    return f1

def findAlpha_BNB(X_train, y_train, X_test, y_test):
    a = 1.0
    best_alpha = 1.0
    best_f1 = 0
    hyperP = []
    for i in range(30):
        f1 = BernoulliNativeBayes(X_train, y_train, X_test, y_test, a)
        hyperP.append([f1, a])
        if (f1 > best_f1):
            best_f1 = f1
            best_alpha = a
        a *= 0.8
    hyperP.sort()
    plotTrainningProcess(hyperP)
    print("The best smoothing parameter alpha = {} \n F1 measure  = {}".format(best_alpha, best_f1))

def decisionTree(X_train, y_train, X_test, y_test, min_sample_split, max_depth):
    clf = DecisionTreeClassifier(min_samples_split=min_sample_split, max_depth=max_depth)
    clf.fit(X_train, y_train)
    predict = []
    for i in range(len(X_test)):
        res = clf.predict([X_test[i]])
        predict.append(res)
    f1 = metrics.f1_score(y_test, predict, average='micro')
    return f1

def findHyperP_DT(X_train, y_train, X_test, y_test):
    best_f1 = 0
    best_max = 1
    best_min_sample = 1.0
    max_depth = 1
    for i in range(10):
        min_sample_split = 1.0
        for i in range(15):
            f1 = decisionTree(X_train, y_train, X_test, y_test, min_sample_split, max_depth)
            if (f1 > best_f1):
                best_f1 = f1
                best_min_sample = min_sample_split
                best_max = max_depth
            min_sample_split *= 0.7
        max_depth += 1
    print("The best F1 measure  = {} \n max_depth = {} \n min_sample_split = {}".format(best_f1, best_max, best_min_sample))

def linearSVM(X_train, y_train, X_test, y_test, C, dual):
    clf = LinearSVC(C=C,dual=dual)
    clf.fit(X_train, y_train)
    predict = []
    for i in range(len(X_test)):
        res = clf.predict([X_test[i]])
        predict.append(res)
    f1 = metrics.f1_score(y_test, predict, average='micro')
    return f1

def GaussianNativeBayes(X_train, y_train, X_test, y_test):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    predict = []
    for i in range(len(X_test)):
        res = clf.predict([X_test[i]])
        predict.append(res)
    f1 = metrics.f1_score(y_test, predict, average='micro')
    return f1

# find hyper parameter of SVM
# c = 1.0
# best_f1 = 0
# bestC = 1.0
# best_dual = False
# for i in range(30):
#     dual = False
#     f1 = linearSVM(trainX, trainY, validX, validY,c, dual)
#     if (f1 > best_f1):
#         best_f1 = f1
#         bestC = c
#         best_dual = dual
#     dual = True
#     f1 = linearSVM(trainX, trainY, validX, validY, c, dual)
#     if (f1 > best_f1):
#         best_f1 = f1
#         bestC = c
#         best_dual = dual
#     c *= 0.8
# print("The F1 measure = {} \nC = {}\nDual = {}".format(best_f1, bestC, best_dual))


def gridSearch_DT(X_train, y_train):
    minSample_range = np.logspace(16, 0, num=15, base=0.8)
    maxDepth_range = range(1, 15)
    param_grid = dict(min_samples_split=minSample_range, max_depth=maxDepth_range)
    DT = DecisionTreeClassifier(min_samples_split=minSample_range, max_depth=maxDepth_range)
    gs = GridSearchCV(DT, param_grid, scoring='f1_micro', cv=3, n_jobs=-1)
    gs.fit(X_train, y_train)
    best_score = gs.best_score_
    best_param = gs.best_params_
    print("The best F1 measure  = {} \nmax_depth = {} \nmin_sample_split = {}".format(best_score, best_param.get('min_samples_split'), best_param.get('max_depth')))

def gridSearch_SVM(X_train, y_train):
    c_range = np.logspace(-5, -20, num=15, base=0.8)
    d_range = [True, False]
    param_grid = dict(C=c_range, dual=d_range)
    svm = LinearSVC(C=c_range, dual=d_range)
    gs = GridSearchCV(svm, param_grid, scoring='f1_micro', n_jobs=-1)
    gs.fit(X_train, y_train)
    best_score = gs.best_score_
    best_param = gs.best_params_
    print("The F1 measure = {} \nC = {}\nDual = {}".format(best_score, best_param.get('C'), best_param.get('dual')))

C = 1.0
array = []
for i in range(30):
    array.append(C)
    C *= 1.2
print(array)

# def findHyperP_DT(X_valid, y_valid):
#     minSample_range = np.logspace(16, 0, num=15, base=0.8)
#     maxDepth_range = range(1,16)
#     param_grid = dict(min_samples_split=minSample_range, max_features=maxDepth_range)
#     DT = DecisionTreeClassifier(min_samples_split=minSample_range, max_features=maxDepth_range)
#     gs = GridSearchCV(DT, param_grid, scoring='f1_micro', n_jobs=-1)
#     gs.fit(X_valid, y_valid)
#     best_score = gs.best_score_
#     best_param = gs.best_params_


# randomClf(trainX, trainY, testX, testY)
# majorityClf(trainX, trainY, testX, testY)

# findAlpha_BNB(trainX, trainY, validX, validY)

# yelpVocab = genVocab("hwk3_datasets/yelp-train.txt")
# imdbVocab = genVocab("hwk3_datasets/IMDB-train.txt")
#
# genVocabFile("hwk3_datasets/yelp-train.txt", "output_datasets/yelp-vocab.txt")
# buildIDFile("hwk3_datasets/yelp-train.txt", yelpVocab, "output_datasets/yelp-train.txt")
# buildIDFile("hwk3_datasets/yelp-test.txt", yelpVocab, "output_datasets/yelp-test.txt")
# buildIDFile("hwk3_datasets/yelp-valid.txt", yelpVocab, "output_datasets/yelp-valid.txt")
#
# genVocabFile("hwk3_datasets/IMDB-train.txt", "output_datasets/IMDB-vocab.txt")
# buildIDFile("hwk3_datasets/IMDB-train.txt", imdbVocab, "output_datasets/IMDB-train.txt")
# buildIDFile("hwk3_datasets/IMDB-test.txt", imdbVocab, "output_datasets/IMDB-test.txt")
# buildIDFile("hwk3_datasets/IMDB-valid.txt", imdbVocab, "output_datasets/IMDB-valid.txt")