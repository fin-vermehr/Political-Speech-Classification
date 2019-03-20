from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVC
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv


def accuracy(C):
    """Compute accuracy given Numpy array confusion matrix C. Returns a floating point value"""
    total = np.sum(C)
    correct = np.sum(np.diag(C))

    return(correct / total)


def recall(C):
    """Compute recall given Numpy array confusion matrix C. Returns a list of floating point values"""
    correct = np.diag(C)
    recall_list = []

    for i in range(len(C)):
        recall_list.append(correct[i] / np.sum(C[:, i]))
    return(recall_list)


def precision(C):
    """Compute precision given Numpy array confusion matrix C. Returns a list of floating point values"""
    correct = np.diag(C)
    precision_list = []

    for i in range(len(C)):
        precision_list.append(correct[i] / np.sum(C[i]))

    return precision_list


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    data = np.load(filename)['arr_0']
    y = data[:, 173]
    X = data[:, : 173]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    accuracy_list = []

    file = open('a1_3.1.csv', 'w')
    writer = csv.writer(file)

    scaler = preprocessing.StandardScaler().fit(X_train)
    SVM_X_train = scaler.transform(X_train)
    SVM_X_test = scaler.transform(X_test)

    # 1. SVC: support vector machine with a linear kernel
    linear_SVM_list = [1]

    linear_SVM = SVC(kernel='linear', max_iter=2000)
    linear_SVM.fit(SVM_X_train, y_train)

    linear_SVM_pred = linear_SVM.predict(SVM_X_test)
    linear_SVM_confusion = confusion_matrix(y_test, linear_SVM_pred)
    linear_SVM_acc = accuracy(linear_SVM_confusion)
    linear_SVM_recall = recall(linear_SVM_confusion)
    linear_SVM_prec = precision(linear_SVM_confusion)
    print(linear_SVM_acc)

    linear_SVM_list.append(linear_SVM_acc)
    linear_SVM_list += linear_SVM_recall
    linear_SVM_list += linear_SVM_prec
    linear_SVM_list += np.ravel(linear_SVM_confusion).tolist()

    print(linear_SVM_confusion)

    writer.writerow(linear_SVM_list)
    accuracy_list.append(linear_SVM_acc)

    # 2. SVC: support vector machine with a radial basis function (γ = 2) kernel.
    rbf_SVM_list = [2]

    rbf_SVM = SVC(kernel='rbf', gamma=2, max_iter=2000)
    rbf_SVM.fit(SVM_X_train, y_train)

    rbf_SVM_pred = rbf_SVM.predict(SVM_X_test)
    rbf_SVM_confusion = confusion_matrix(y_test, rbf_SVM_pred)
    rbf_SVM_acc = accuracy(rbf_SVM_confusion)
    rbf_SVM_recall = recall(rbf_SVM_confusion)
    rbf_SVM_prec = precision(rbf_SVM_confusion)

    rbf_SVM_list.append(rbf_SVM_acc)
    rbf_SVM_list += rbf_SVM_recall
    rbf_SVM_list += rbf_SVM_prec
    rbf_SVM_list += np.ravel(rbf_SVM_confusion).tolist()

    print(rbf_SVM_confusion)

    writer.writerow(rbf_SVM_list)
    accuracy_list.append(rbf_SVM_acc)

    # 3. RandomForestClassifier: with a maximum depth of 5, and 10 estimators.
    RFC_list = [3]

    RFC = RandomForestClassifier(n_estimators=10, max_depth=5)
    RFC.fit(X_train, y_train)

    RFC_pred = RFC.predict(X_test)
    RFC_confusion = confusion_matrix(y_test, RFC_pred)
    RFC_acc = accuracy(RFC_confusion)
    RFC_recall = recall(RFC_confusion)
    RFC_prec = precision(RFC_confusion)

    RFC_list.append(RFC_acc)
    RFC_list += RFC_recall
    RFC_list += RFC_prec
    RFC_list += np.ravel(RFC_confusion).tolist()

    writer.writerow(RFC_list)
    accuracy_list.append(RFC_acc)

    # 4. MLPClassifier: A feed-forward neural network, with α = 0.05.
    MLP_list = [4]

    MLP = MLPClassifier(alpha=0.05)
    MLP.fit(SVM_X_train, y_train)

    MLP_pred = MLP.predict(SVM_X_test)
    MLP_confusion = confusion_matrix(y_test, MLP_pred)
    MLP_acc = accuracy(MLP_confusion)
    MLP_recall = recall(MLP_confusion)
    MLP_prec = precision(MLP_confusion)

    MLP_list.append(MLP_acc)
    MLP_list += MLP_recall
    MLP_list += MLP_prec
    MLP_list += np.ravel(MLP_confusion).tolist()

    writer.writerow(MLP_list)
    accuracy_list.append(MLP_acc)

    # 5. AdaBoostClassifier: with the default hyper-parameters.
    ABC_list = [5]

    ABC = AdaBoostClassifier()
    ABC.fit(X_train, y_train)

    ABC_pred = ABC.predict(X_test)
    ABC_confusion = confusion_matrix(y_test, ABC_pred)
    ABC_acc = accuracy(ABC_confusion)
    ABC_recall = recall(ABC_confusion)
    ABC_prec = precision(ABC_confusion)

    ABC_list.append(ABC_acc)
    ABC_list += ABC_recall
    ABC_list += ABC_prec
    ABC_list += np.ravel(ABC_confusion).tolist()

    writer.writerow(ABC_list)
    accuracy_list.append(ABC_acc)

    # Calculate iBest

    max_acc = max(accuracy_list)
    iBest = accuracy_list.index(max_acc) + 1

    return(X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test, iBest):
    """This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
    """
    training_sizes = [1000, 5000, 10000, 15000, 20000]

    file = open('a1_3.2.csv', 'w')
    writer = csv.writer(file)

    index_list = list(range(len(y_train)))
    accuracy_list = []

    models = [SVC(kernel='linear', max_iter=100),
              SVC(kernel='rbf', gamma=2, max_iter=100),
              RandomForestClassifier(n_estimators=10, max_depth=5),
              MLPClassifier(alpha=0.05), AdaBoostClassifier()]

    best_model = models[iBest - 1]

    for size in training_sizes:

        indices = np.random.choice(index_list, size)
        r_X_train = np.asarray([X_train[i] for i in indices])
        r_y_train = np.asarray([y_train[i] for i in indices])

        if size == 1000:
            X_1k = r_X_train
            y_1k = r_y_train

        model = best_model
        model.fit(r_X_train, r_y_train)

        model_pred = model.predict(X_test)
        model_confusion = confusion_matrix(y_test, model_pred)
        accuracy_list.append(accuracy(model_confusion))

    writer.writerow(accuracy_list)
    file.close()

    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    file = open('a1_3.3.csv', 'w')
    writer = csv.writer(file)

    models = [SVC(kernel='linear', max_iter=100),
              SVC(kernel='rbf', gamma=2, max_iter=100),
              RandomForestClassifier(n_estimators=10, max_depth=5),
              MLPClassifier(alpha=0.05), AdaBoostClassifier()]

    model = models[iBest - 1]

    accuracy_list = []

    for dataset in [(X_1k, y_1k), (X_train, y_train)]:
        for k in [5, 10, 20, 30, 40, 50]:
            selector = SelectKBest(f_classif, k)
            selector.fit(X=dataset[0], y=dataset[1])
            pp = selector.pvalues_
            mask = selector.get_support()
            print([i for i, x in enumerate(mask) if x])

            if dataset == (X_train, y_train):
                output = [k]
                for i in pp[mask]:
                    output.append(i)

                writer.writerow(output)

            if k == 5:
                X_train_new = selector.transform(dataset[0])
                X_test_new = selector.transform(X_test)

                model.fit(X_train_new, dataset[1])

                model_pred = model.predict(X_test_new)
                model_confusion = confusion_matrix(y_test, model_pred)
                accuracy_list.append(accuracy(model_confusion))

    writer.writerow(accuracy_list)
    file.close()


def class34(filename, i):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
    '''
    file = open('a1_3.4.csv', 'w')
    writer = csv.writer(file)

    models = [SVC(kernel='linear', max_iter=2000),
              SVC(kernel='rbf', gamma=2, max_iter=2000),
              RandomForestClassifier(n_estimators=10, max_depth=5),
              MLPClassifier(alpha=0.05), AdaBoostClassifier()]

    linear_acc = []
    rbf_acc = []
    RFC_acc = []
    MLP_acc = []
    ABC_acc = []

    significance_list = []

    data = np.load(filename)['arr_0']
    y = data[:, 173]
    X = data[:, : 173]
    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X):
        accuracy_list = []
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        for index in range(len(models)):
            model = models[index]
            model.fit(X_train, y_train)

            model_pred = model.predict(X_test)
            model_confusion = confusion_matrix(y_test, model_pred)

            accuracy_list.append(accuracy(model_confusion))

            if index == 0:
                linear_acc.append(accuracy(model_confusion))
            elif index == 1:
                rbf_acc.append(accuracy(model_confusion))
            elif index == 2:
                RFC_acc.append(accuracy(model_confusion))
            elif index == 3:
                MLP_acc.append(accuracy(model_confusion))
            else:
                ABC_acc.append(accuracy(model_confusion))

        writer.writerow(accuracy_list)

    S1 = stats.ttest_rel(ABC_acc, linear_acc)
    S2 = stats.ttest_rel(ABC_acc, rbf_acc)
    S3 = stats.ttest_rel(ABC_acc, RFC_acc)
    S4 = stats.ttest_rel(ABC_acc, MLP_acc)

    significance_list.append(S1[1])
    significance_list.append(S2[1])
    significance_list.append(S3[1])
    significance_list.append(S4[1])

    writer.writerow(significance_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(args.input, iBest)
