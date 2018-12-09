import csv
import random

# This script will fill the missing data by sample mean, and do 80-20 splits
#
# This script will generate 5 CandC-train files and 5 CandC-test files in directory Datasets
#
# Note: Do not run this sciprt unless the CandC-train files and CandC-test files are missing

with open("Datasets/communities.csv") as csvFile:
    trainDataSet = csv.reader(csvFile, delimiter=',')
    data = []
    for row in trainDataSet:
        data.append(row)

with open("Datasets/stat.csv") as csvFile:
    trainDataSet = csv.reader(csvFile, delimiter=',')
    stat = []
    for row in trainDataSet:
        stat.append(row)

for i in range(len(data)):
    for j in range(5, len(data[0])) : # The first five columns are non-predictive
        if (data[i][j] == "?"):
            data[i][j] = stat[j-5][3]

# now we shuffle data and do 80-20 splits

boundry = int(len(data)*0.8)

for i in range(1, 6):
    random.shuffle(data)
    train_data = data[:boundry]
    test_data = data[boundry:]

    with open('Datasets/CandC-train' + str(i) + '.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerows(train_data)

    with open('Datasets/CandC-test' + str(i) + '.csv', 'w') as csvfile:
        csvWriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvWriter.writerows(test_data)