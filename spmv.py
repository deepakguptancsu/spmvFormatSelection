import numpy
import os
import scipy.sparse as sp
from scipy.io.mmio import mmread
import time
import sys
from tqdm import tqdm
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

##
##
##
##Function to calculate execution time of a format on given 
##input matrix and change the label if execution time is less
##than minimum execution time
##
##
def checkExeTime(matData, conversionTime, minExeTime, currentLabel, checkLabel):
    defaultVector = numpy.full((matData.shape[1],1), 1, float)

    startTime = time.time()
    #running multiplication for 50 times
    for i in range(1,51):
        finalVector = matData * defaultVector
    endTime = time.time()
    
    exeTime = endTime - startTime + conversionTime

    if(exeTime < minExeTime):
        minExeTime = exeTime
        currentLabel = checkLabel
    
    return (minExeTime, currentLabel)

##
##
##function to find which format is best for the given sparse array
##in terms of execution time
##
##format -> classLabel
##COO -> 1, CSR -> 2, CSC->3, BSR -> 4, DIA -> 5, DOK -> 6, LIL -> 7
##
##
##
def findLabel(matDataOriginal):
    minExeTime = sys.float_info.max
    finalLabel = -1

    #coo to coo conversion time is 0
    minExeTime,finalLabel = checkExeTime(matDataOriginal, 0, minExeTime, finalLabel, 1)
    
    startTime = time.time()
    matData = matDataOriginal.tocsr()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 2)

    startTime = time.time()
    matData = matDataOriginal.tocsc()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 3)

    startTime = time.time()
    matData = matDataOriginal.tobsr()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 4)

#    startTime = time.time()
#    matData = matDataOriginal.todia()
#    endTime = time.time()
#    conversionTime = endTime - startTime
#    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 5)

    startTime = time.time()
    matData = matDataOriginal.todok()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 6)

    startTime = time.time()
    matData = matDataOriginal.tolil()
    endTime = time.time()
    conversionTime = endTime - startTime
    minExeTime,finalLabel = checkExeTime(matData, conversionTime, minExeTime, finalLabel, 7)

    return finalLabel

##
##
##function to calculate all attributes of a sparse matrix
##required to train model
##
##
def calAttributes(matData):
    if(sp.isspmatrix_coo(matData) == False):
        matData = sp.coo_matrix(matData)
    
    #variable list to store all attributes of sparse matrix
    #required to train model 
    attributeList = []

    #writing filename in first column of attribute list
    #attributeList.append(fileName)

    #writing number of rows and columns in list
    numRows = matData.shape[0]
    numCol = matData.shape[1]
    attributeList.append(numRows)
    attributeList.append(numCol)

    #writing number of non-zeros in list
    nnz = matData.count_nonzero()
    attributeList.append(nnz)

    #writing number of diagonals in list
    Ndiags = numCol + numRows - 1
    attributeList.append(Ndiags)

    #writing NTdiags ratio to list


    #attributes for nnzs per row
    rowArr = matData.row
    nnzRows = numpy.full(matData.shape[0], 0, float)

    for i in range(rowArr.size):
       nnzRows[rowArr[i]] += 1

    aver_RD = numpy.mean(nnzRows)
    max_RD = numpy.max(nnzRows)
    min_RD = numpy.min(nnzRows)
    dev_RD = numpy.std(nnzRows)

    attributeList.append(aver_RD)
    attributeList.append(max_RD)
    attributeList.append(min_RD)
    attributeList.append(dev_RD)

    #attributes for nnzs per col
    colArr = matData.col
    nnzCol = numpy.full(matData.shape[1], 0, float)

    for i in range(colArr.size):
        nnzCol[colArr[i]] += 1

    aver_CD = numpy.mean(nnzCol)
    max_CD = numpy.max(nnzCol)
    min_CD = numpy.min(nnzCol)
    dev_CD = numpy.std(nnzCol)

    attributeList.append(aver_CD)
    attributeList.append(max_CD)
    attributeList.append(min_CD)
    attributeList.append(dev_CD)

    #calculating ER_DIA
    #matDia = matData.todia()
    #matDiaData = matDia.data
    #ER_DIA = (numpy.count_nonzero(matDiaData))/(matDiaData.shape[0]*matDiaData.shape[1])
    #attributeList.append(ER_DIA)

    #calculating ER_RD
    ER_RD = nnz/(max_RD*numRows)
    attributeList.append(ER_RD)

    #calculating ER_CD
    ER_CD = nnz/(numCol*max_CD)
    attributeList.append(ER_CD)

    #calculating row_bounce and col_bounce
    diffAdjNnzRows = numpy.full(nnzRows.size - 1, 0, float)
    for i in range(1,nnzRows.size):
        diffAdjNnzRows[i-1] = numpy.absolute(nnzRows[i] - nnzRows[i-1])

    row_bounce = numpy.mean(diffAdjNnzRows)

    diffAdjNnzCols = numpy.full(nnzCol.size - 1, 0, float)
    for i in range(1,nnzCol.size):
        diffAdjNnzCols[i-1] = numpy.absolute(nnzCol[i] - nnzCol[i-1])

    col_bounce = numpy.mean(diffAdjNnzCols)

    attributeList.append(row_bounce)
    attributeList.append(col_bounce)

    #calculating density of matrix
    densityOfMatrix = (matData.count_nonzero())/((matData.shape[0])*(matData.shape[1]))
    attributeList.append(densityOfMatrix)

    #calculating normalized variation of nnz per row
    nnzRowsNormalised = (nnzRows-min_RD)/max_RD
    cv = numpy.var(nnzRowsNormalised)
    attributeList.append(cv)

    #caluculating max_mu
    max_mu = max_RD - aver_RD
    attributeList.append(max_mu)

    #calculating number of non_zero blocks



    #calculating average number of NZ neighbors of an element
    """
    dict = {}

    for i in range(0,nnz):
        keyStr = "" + numpy.array2string(rowArr[i]) + numpy.array2string(colArr[i])
        dict[keyStr] = 1

    neighbour = []

    for i in range(0,numRows):
        for j in range(0,numCol):
            countNum = 0
            if (str(i-1)+str(j-1)) in dict:
                countNum = countNum + 1
            if (str(i-1)+str(j)) in dict:
                countNum = countNum + 1
            if (str(i-1)+str(j+1)) in dict:
                countNum = countNum + 1
            if (str(i)+str(j-1)) in dict:
                countNum = countNum + 1
            if (str(i)+str(j+1)) in dict:
                countNum = countNum + 1
            if (str(i+1)+str(j-1)) in dict:
                countNum = countNum + 1
            if (str(i+1)+str(j)) in dict:
                countNum = countNum + 1
            if (str(i+1)+str(j+1)) in dict:
                countNum = countNum + 1
            
            if(countNum != 0):
                neighbour.append(countNum)
            

    neighbour = numpy.asarray(neighbour)
    mean_neighbour = numpy.sum(neighbour)
    mean_neighbour = mean_neighbour/(numRows*numCol)

    attributeList.append(mean_neighbour)
    """

    #find out which format is best for the given sparse array
    #in terms of execution time
    formatLabel = findLabel(matData)
    attributeList.append(formatLabel)

    return attributeList

##
##
##function which reads input matrices and creates
##training data and dumps it in spmvData.csv
##
##
def createTrainingData():
    finalAttributeList = []
    inputDir = "./sparseMatrixdataSet/"


    for fileName in tqdm(os.listdir(inputDir)):
        fileNameWithPath = inputDir + fileName
        matData = mmread(fileNameWithPath)
        if((sp.isspmatrix_coo(matData) == True) or ((sp.isspmatrix_coo(matData) == False) and (matData.shape[1] != 1))):
            attributeList = calAttributes(matData)
            finalAttributeList.append(attributeList)
        else:
            print(fileName+" is not a sparse matrix.\n")

    finalArr = numpy.asarray(finalAttributeList)
    numpy.savetxt("spmvData.csv", finalArr, fmt='%f', delimiter=",")

    print("Training data is stored in spmvData.csv\n")


##
##
##ignore unnecessary warnings of scipy
##
##
##
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


##
##
##main execution starts from here
##
##
input1 = input("Press 1: To create new training data/spmvData.csv \nPress 2: To use existing training data/spmvData.csv\n")
option = int(input1)
if option == 1:
    createTrainingData()
elif option == 2:
    print("using existing spmvData.csv\n")
else:
    print("Invalid Option\n")


print("Training the model using spmvData.csv\n")

# load data
dataset = loadtxt('spmvData.csv', delimiter=",")
numOfCol = dataset.shape[1]

# split data into X and y
X = dataset[:,0:numOfCol-2]
Y = dataset[:,numOfCol-1]

kf = KFold(n_splits=5)
kf.get_n_splits(X)

#creating accuracy array for storing accuracy from each fold
#of 5 fold cross validation
accracyArr = numpy.full(5, 0, float)
accuracyCounter = 0

for trainIndex, testIndex in kf.split(X):
    #print("TRAIN: ", trainIndex, "TEST: ", testIndex)
    xTrain, xTest = X[trainIndex], X[testIndex]
    yTrain, yTest = Y[trainIndex], Y[testIndex]

    # fit model on training data
    model = XGBClassifier()
    model.fit(xTrain, yTrain)

    # make predictions for test data
    yPred = model.predict(xTest)
    predictions = [round(value) for value in yPred]

    # evaluate predictions
    accracyArr[accuracyCounter] = accuracy_score(yTest, predictions)
    accuracyCounter = accuracyCounter + 1

#taking mean of accuracy of each fold from 5 fold cross validation
accuracyMean = numpy.mean(accracyArr)
print("Accuracy: %.2f%%" % (accuracyMean * 100.0))