##Logistic Regressoin on Computational Features to predict Semantic Features 
##of liver lesions 
#fjeg

import numpy as np 
import pandas as pd
import sys 
import operator 
import math

from sklearn import linear_model 
from sklearn import cross_validation
from sklearn import metrics
from joblib import Parallel, delayed

NUM_CASES = 79
NUM_SEMANTIC_FEAT = 30 
NUM_COMP_FEAT = 431 


def main(): 
	##load and process data 
	flag = "L1"
	compFeatures = open('../featurized/computational_features.csv', 'r')
	comp_uids = open('../featurized/comp_uids.csv', 'r')
	lesionFeaturesFile = open('../featurized/lesion_features.txt', 'r')
	allSemFeatures = open('../featurized/semantic_feature_list.txt', 'r')
	cfMatrix = np.genfromtxt(compFeatures, delimiter=",") #79x431
	semFeatureMatrix = constructOutputMatrix(comp_uids, lesionFeaturesFile, allSemFeatures) #79x30
	#parallelize(cfMatrix, semFeatureMatrix, flag)
	#fitModel(cfMatrix, semFeatureMatrix, flag)


def constructOutputMatrix(comp_uids, lesionFeaturesFile, allSemFeatures): 
	##create map from image id to index  
	idMap = {}
	i = 0
	for line in comp_uids: 
		idMap[line.strip()] = i
		i += 1

	##create semantic feature map (key: feature, value: index of feature)
	semFeatures = allSemFeatures.readline().strip().split()
	semFeatureMap = {}
	for i in xrange(len(semFeatures)): 
		semFeatureMap[semFeatures[i]] = i

	#process lesion features and create output matrix 
	semFeatureMatrix = np.zeros((NUM_CASES, NUM_SEMANTIC_FEAT), dtype=np.int)
	curCase = 0
	for line in lesionFeaturesFile: 
		elem = line.strip().split()
		if len(elem) == 1: 
			curCase = idMap[elem[0]]
		elif len(elem) != 0: 
			index = semFeatureMap.get(elem[1], -1)
			if index != -1: semFeatureMatrix[curCase, index] = 1
			#semFeatureMatrix[curCase, semFeatureMap.get(elem[1], 0)]
	print semFeatureMatrix
	semdf = pd.DataFrame(semFeatureMatrix)
	print semdf
	# for i in xrange(len(semFeatureMatrix[0])): 
	# 	np.where(semFeatureMatrix[i])
	return semFeatureMatrix


def parallelize(cfMatrix, semFeatureMatrix, flag): 
	loocv = cross_validation.LeaveOneOut(NUM_CASES)
	logreg = linear_model.LogisticRegression() 
	#aucList, mcrList = [], []
	auc_mcr_list = []
	#Y_true, Y_scores = [], []
	#fitModel(auc_mcr_list, loocv, logreg, cfMatrix, semFeatureMatrix, flag, 1)
	auc_mcr_list = Parallel(n_jobs=-1)(delayed(fitModel)(loocv, logreg, cfMatrix, semFeatureMatrix, flag, i) 
		for i in xrange(len(semFeatureMatrix[0])))
	print auc_mcr_list


def fitModel(loocv, logreg, cfMatrix, semFeatureMatrix, flag, i): 
	# loocv = cross_validation.LeaveOneOut(NUM_CASES)
	# logreg = linear_model.LogisticRegression() 
	# #print cfMatrix.shape 
	# #print semFeatureMatrix.shape
	# #errorLambdaMatrix = [[] for _ in range(NUM_CASES)]
	# aucList, mcrList = [], []

	# #for train_index, test_index in loocv: 
	# #Parallel(n_jobs=-1)
	# #for i in xrange(len(semFeatureMatrix[0])): 
	# 	#mean_auc = 0.0
	Y_true, Y_scores = [], []
	print i 
	mcr = np.zeros(len(loocv))
		#print "mcr: " + str(mcr.shape)
		#if i != 24: continue
	for train_index, test_index in loocv:
			#print test_index
			#intern_loocv = cross_validation.LeaveOneOut(len(train_index))
		intern_kf = cross_validation.KFold(len(train_index), n_folds=10)
		Y = np.delete(semFeatureMatrix[:,i], test_index, 0) #check to make sure this works
		X = np.delete(cfMatrix, test_index[0], 0)
		logreg = linear_model.LogisticRegression(penalty='l1', solver='liblinear')
		if flag == "L1": bestParams = tuneParams(X, Y, intern_kf, logreg)
		logreg.set_params(C=bestParams[0])
		threshold = bestParams[1]
		#threshold = tuneThreshold(X, Y, intern_kf, logreg)
		Y_test = semFeatureMatrix[test_index[0], i]
		Y_pred = predict(semFeatureMatrix, cfMatrix, test_index, i, logreg)
		if (Y_pred > threshold) == Y_test: mcr[test_index] = 0
		else: mcr[test_index] = 1 
			#errorLambdaMatrix[test_index[0], i] = paramErrorTuple
			#errorLambdaMatrix[test_index[0]].append(paramErrorTuple)
			#print "Y_test: " + str(Y_test) + " Y_pred: " + str(Y_pred)
		Y_true.append(Y_test)
		Y_scores.append(Y_pred)
			#break
		#print len(Y_scores)
	auc = metrics.roc_auc_score(Y_true, Y_scores)
	print (i, (auc, np.mean(mcr)))
	return (i, (auc, np.mean(mcr)))
	#auc_mcr_list.append((i, (auc, np.mean(mcr))))


def tuneParams(X, Y, intern_kf, logreg): 
	C = np.logspace(0,4,num=10,base=2) ##change values of C
	thresholds = np.array([np.linspace(0,1,num=10)])
	Y = np.transpose(np.array([Y]))
	#Y_pred = np.zeros(Y.shape)
	#Y_true = np.zeros(Y.shape)
	bestParamComboArray = np.zeros((10, 3))
	k = 0
	for train, test in intern_kf: 
		#print "test: " + str(test)
		X_train, Y_train = X[train], Y[train]
		X_test, Y_test = X[test], Y[test]
		#Y_pred = np.zeros(Y_test.shape)
		paramCombArray = np.zeros((10, 3))
		for i in xrange(C.shape[0]): 
			logreg.set_params(C=C[i])
			logreg.fit(X_train, np.ravel(Y_train))
			Y_pred = np.transpose(np.array([logreg.predict_proba(X_test)[:,1]]))
			thresholdGrid = Y_pred > thresholds
			misclassGrid = Y_test != thresholdGrid 
			mcrs = np.mean(misclassGrid, axis=0)
			fillParamCombArray(paramCombArray, i, C[i], thresholds[0,np.argmin(mcrs)], np.amin(mcrs))
		indexMinMCR = paramCombArray.argmin(0)[2]

		#storing best combination at end of 1 fold
		fillParamCombArray(bestParamComboArray, k, paramCombArray[indexMinMCR, 0], paramCombArray[indexMinMCR, 1], paramCombArray[indexMinMCR, 2])
		k += 1
	indexBestCombo = bestParamComboArray.argmin(0)[2]
	return bestParamComboArray[indexBestCombo]


def fillParamCombArray(paramCombArray, row, C, threshold, mcr): 
	paramCombArray[row,0] = C
	paramCombArray[row,1] = threshold
	paramCombArray[row,2] = mcr

# def tuneThreshold(X, Y, intern_kf, logreg): 
# 	#t_prep = np.linspace(0,1,num=10)
# 	thresholds = np.array([np.linspace(0,1,num=100)])
# 	Y = np.transpose(np.array([Y]))
# 	Y_pred = np.zeros(Y.shape)
# 	Y_true = np.zeros(Y.shape)
# 	for train, test in intern_kf: 
# 		X_train, Y_train = X[train], Y[train]
# 		X_test, Y_test = X[test], Y[test]
# 		logreg.fit(X_train, np.ravel(Y_train))
# 		Y_pred[test] = logreg.predict_proba(X_test)[0,1]
# 		Y_true[test] = Y_test

# 		# Y_train = Y[train]
# 		# Y_test = Y[test]
# 	thresholdGrid = Y_pred > thresholds
# 	misclassGrid = Y_true != thresholdGrid
# 	means = np.mean(misclassGrid, axis=0)
# 	return thresholds[0,np.argmin(means)]
# 	# print means
# 	# print thresholds
# 	# print means.shape
# 	# print thresholdGrid.shape
# 	# print "thresholds: " + str(thresholds.shape)
# 	# print "Y.shape: " + str(Y.shape) 


# def tuneHyperparam(Y, X): 
# 	logreg = linear_model.LogisticRegression(penalty='l1', solver='liblinear')
# 	C = np.logspace(1,100,num=10) ##change values of C
# 	# #print C.shape[0]
# 	# grid = np.zeros((10,))
# 	# # #print grid.shape
# 	# for i in xrange(C.shape[0]):
# 	# 	logreg.set_params(C=C[i])
# 	#  	#logreg.fit(cfMatrix, semFeatureMatrix[:,curFeature])
# 	#  	scores = cross_validation.cross_val_score(logreg, X, y=Y, scoring='accuracy', cv=5)
# 	#  	grid[i] = scores.mean()

# 	# 	#print "first accuracy score"
# 	# 	#grid[i] = metrics.accuracy_score(semFeatureMatrix[:,curFeature], predicted)
# 	# 	#print metrics.accuracy_score(semFeatureMatrix[:,curFeature], predicted)
# 	# print grid
# 	return linear_model.LogisticRegressionCV(Cs=C, penalty='l1', solver='liblinear', n_jobs=-1)


def predict(semFeatureMatrix, cfMatrix, test_index, curFeature, logreg):
	Y_train = np.delete(semFeatureMatrix[:,curFeature], test_index, 0)
	#print Y_train.shape
	#Y_train = semFeatureMatrix[0:test_index[0], curFeature] + semFeatureMatrix[test_index[0]:, curFeature] #training 
	X_train = np.delete(cfMatrix, test_index[0], 0)
	X_test = cfMatrix[test_index[0],:]
	logreg.fit(X_train, Y_train)
	Y_pred = logreg.predict_proba(X_test)
	#print "classes: " + str(logreg.classes_)
	#print "Y_pred: " + str(Y_pred)
	#print "Y_pred shape: " + str(Y_pred.shape)
 	#residual = abs(Y_test - Y_pred[0,1])
	##MSE = metrics.mean_squared_error(Y_test, Y_pred[0])
	#lambd = logreg.get_params()
	#return (lambd, residual), Y_test, Y_pred[0,1]
	return Y_pred[0,1]


	#print cfMatrix.shape, semFeatureMatrix.shape
	#scores = cross_validation.cross_val_score(logreg, cfMatrix, semFeatureMatrix[:,0], cv=loocv)
	##ROC with cross-validation 
	#6mean_tpr = 0.0
	#predicted = cross_validation.cross_val_predict(clf, cfMatrix, semFeatureMatrix[:,0], cv=loocv)
	# metrics.accuracy_score(semFeatureMatrix[:,0], predicted)
	#print scores
	#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
	

	# logreg = linear_model.LogisticRegression()
	# #blah = np.zeros((79,), dtype=np.int)
	# logreg.fit(cfMatrix, semFeatureMatrix[:,0])
	# loocv = cross_validation.LeaveOneOut(NUM_COMP_FEAT) 


if __name__ == '__main__':
    main()

