
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# Name: Parvez Golam
# Date: 20/05/2021
# DEscription - Classifiers comparision for classification of 
#                Boolean satisfiability (SAT) problem
# Classifiers used: Naive Bayes, Ridge Classifier, K-Nearest Neighbours, 
#                   Decision Tree, Bagging, Random Forest, Suport Vector Machine.
#-----------------------------------------------------------------------------
#-----------------------------------------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, accuracy_score, hamming_loss, jaccard_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold, learning_curve
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# constants
BASIC = "Basic Model"
NB = "Naive Bayes"
RIDGE = "Ridge Classifier"
KNN = 'Knn'
DECISION_TREE = "Decision tree"
BAGGING = "Bagging"
RANDOM_FOREST = "Random Forest"
SVM = "SVM"
CHECK = "check_dataset"

def check_dataset(dataset):
    # Function which runs a basic 
    # check of  different properties of dataset

    print("\nDataset Information:\n")
    print(dataset.describe())

    print("\nDataset datatypes:")
    print(dataset.dtypes.value_counts())

    print("\nClass Labels:")
    print(dataset.ebglucose_solved.value_counts())

    print("\nMissing Values:")
    print(dataset.isnull().sum().sort_values(ascending=False))

    # Check data distributionS (checking first 10 to get some idea)
    for i in range(11):
        data = dataset.iloc[:,i]
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.25, .75)})

        sns.boxplot(x=data, ax=ax_box)
        sns.histplot(data, ax=ax_hist)

        ax_box.set(xlabel='')
        plt.show()

def runBasicClassifier(classifier, feature_train, label_train, feature_test): 
    '''
    # Function to fit a classifier given in 'classifier' on training data
    # and return predicted results for test data
    '''
     
    classifier.fit(feature_train, label_train)
    predictedResults = classifier.predict(feature_test)

    return predictedResults

def classsifierEvaluation(clf, predictedResults, X_test, label_test, title):
    '''
    Function that producces different evaluation metric
    (accuracy, confusion matrix) for a classifier in "clf"
    '''

    # Accuracy
    accuracy = accuracy_score(predictedResults, label_test)
    print ('Test Accuracy Score: %.3f' %accuracy)

    # Classification Report
    cr = classification_report(label_test, predictedResults)
    print("Classification Report:\n")
    print(cr)

    # Confusion Matrix
    cm = confusion_matrix(label_test, predictedResults)
    print("Confusion Matrix:\n", pd.DataFrame(cm))
    plot_confusion_matrix(clf, X_test, label_test )
    plt.title(title)
    plt.show()

def basic_model(dataset):
    '''
    # Function to execute the basic model with KNN and Random Forest
    # with out optimising the parameters and compare the results
    '''

    # labels and features
    labels = dataset.iloc[:,1]
    dataset = dataset.drop(dataset.columns[[1]], axis=1)

    # data split -70% training and 30% testing 
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.30, random_state=111)
    
    # knn classifier with the k-value 3(random choice)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn_pred_results = runBasicClassifier(knn, X_train, y_train, X_test)
    print("\nKNN classifier:")
    classsifierEvaluation(knn, knn_pred_results, X_test, y_test, "Confusion Matrix - KNN")

    # Random forest with n_estimators 100(default choice)
    rf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=111) 
    rf_pred_results = runBasicClassifier(rf, X_train, y_train, X_test)
    print("\nRandom Forest:")
    classsifierEvaluation(rf, rf_pred_results, X_test, y_test, "Confusion Matrix - Random Forest")

def data_preprocessing(dataset, model):
    '''
    # Function that pre process dataset
    # 1) Feature Label extraction
    # 2) Feature scaling
    # 3) Outlier detection
    '''

    # labels and features
    y = dataset.iloc[:,1]
    X = dataset.drop(dataset.columns[[1]], axis=1)

    # remove 'INSTANCE_ID'
    X = X.drop(dataset.columns[[0]], axis=1)

    # Feature Normalization
    scaler = PowerTransformer() if model == NB else StandardScaler() 
    X = scaler.fit_transform(X)
    y = y.to_numpy()

    # outlier detection and removal
    # lowest 0.5%  data removed as outlier 
    out = LocalOutlierFactor(n_neighbors=20)
    out.fit_predict(X) 
    lof = out.negative_outlier_factor_
    thresh = np.quantile(lof, 0.005)
    index = np.where(lof>thresh)
    X_selected = X[index]
    y_selected = y[index]

    return X, y, X_selected, y_selected

def fit_classifier(fs, clf, params, X_selected, y_selected):
    '''
    # Function which fits classifier with hyper parameter optimization
    # for robust evaluation an returns Fitted results
    '''

    # Pipe line 
    pipe_lr = Pipeline([ ('fs', fs), ('clf', clf )])

    # Repeated Stratified Cross validation
    cv_method = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=111)

    # Grid search for hyper parameter optimization
    gs = GridSearchCV(pipe_lr, param_grid=params, cv=cv_method)

    # Fit classifier without the outliers 
    results = gs.fit(X_selected, y_selected)

    print('Best Mean Accuracy: %.3f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)

    return results

def robust_evaluation(best_estimator, X, y):
    '''
    #Function that preditcs the class level base on best estimator 
    # and prints different evaluation matrices
    '''

    # Cross validated prediction from the best estimator 
    y_pred = cross_val_predict(best_estimator, X, y, 
                                cv= StratifiedKFold(n_splits=5, 
                                                    random_state=111, 
                                                    shuffle=True))
    
    print("Classification Report:\n", classification_report(y, y_pred))

    print("\nConfusion Matrix:\n", pd.DataFrame(confusion_matrix(y, y_pred)))

    print("\nAccuracy:", accuracy_score(y_pred, y))

    print("\nHamming loss:", hamming_loss(y, y_pred))

    print("\nJaccard Score for levels:\n", jaccard_score( y, y_pred, average=None))

def get_learning_curve(best_estimator, X, y, title):
    '''
    # Function that produces the learning curves based on the best estimator 
    generated by grid search 'best_estimator'
    '''

    # get the train test scores
    train_sizes, train_scores, test_scores = learning_curve(best_estimator, X, y,  
                                                            train_sizes= np.linspace(.05, 1.0, 15), 
                                                            cv=StratifiedKFold(n_splits=5, 
                                                                              random_state=111, 
                                                                              shuffle=True))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # plot the learning curve
    plt.style.use("ggplot")
    plt.plot(train_sizes, train_scores_mean, marker='.', label="Training score")
    plt.plot(train_sizes, test_scores_mean, marker='.', label="Cross-validation score")
    plt.legend()
    plt.title(title)
    ylim=(0.0, 1.2)
    plt.ylim(ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.show()

def robust_classification(dataset, model):
    '''
    # Function that perform robust classification for different classifiers
    based on 'model'---
    " Naive Bayes, Ridge Classifier, K-Nearest Neighbours, 
    Decision Tree, Bagging, Random Forest, Suport Vector Machines" 
    '''

    # Data preprocessing
    X, y, X_selected, y_selected = data_preprocessing(dataset, model)

    if model == NB:  #  Naive Bayes

        print("\nRobust Evaluation--Naive Bayes:\n")

        # Hyper Parameters 
        params = {
                'fs__n_components':list(range(65,126,5)),
                'clf__var_smoothing': np.logspace(0,-3, num=5)
                }

        # model fit
        results_nb = fit_classifier(PCA(), GaussianNB(), params, X_selected, y_selected)

        # Plot of NB CV scores for comparision
        results = pd.DataFrame(results_nb.cv_results_['params'])
        results['test_score'] = results_nb.cv_results_['mean_test_score']
        plt.style.use("ggplot")
        temp_avg = results.groupby('clf__var_smoothing').agg({'test_score': 'max'})
        plt.plot(temp_avg, marker = '.')    
        plt.xlabel('Var. Smoothing')
        plt.ylabel("Mean CV Score")
        plt.title("NB Performance Comparison")
        plt.show()

        # performance metrices for Naive Bayes based on the best estimator
        # selected in Grid search
        robust_evaluation(results_nb.best_estimator_, X, y)

        # Learing curve NB
        title = "Learning Curve - NB"
        get_learning_curve(results_nb.best_estimator_, X, y, title)

    elif model == RIDGE: # Ridge classifier

        print("\nRobust Evaluation--Ridge classifier:\n")

        # Hyper Parameters 
        params = {
                'fs__n_components':list(range(65,126,5)),
                'clf__alpha': np.logspace(0,-4, num=5)
                }

        # model fit
        results_ridge = fit_classifier(PCA(), RidgeClassifier(), params, X_selected, y_selected)

        # Plot ridge CV scores for comparision
        results = pd.DataFrame(results_ridge.cv_results_['params'])
        results['test_score'] = results_ridge.cv_results_['mean_test_score']
        plt.style.use("ggplot")
        temp_avg = results.groupby('clf__alpha').agg({'test_score': 'max'})
        plt.plot(temp_avg, marker = '.')    
        plt.xlabel('Alpha')
        plt.ylabel("Mean CV Score")
        plt.title("Ridge classifier Performance Comparison")
        plt.show()

        # performance metrices for Ridge classifier based on the best estimator
        # selected in Grid search
        robust_evaluation(results_ridge.best_estimator_, X, y)

        # Learing curve Ridge 
        title = "Learning Curve - Ridge classifier"
        get_learning_curve(results_ridge.best_estimator_, X, y, title)
    
    elif model == KNN: # Knn

        print("\nRobust Evaluation--KNN:\n")

        # Hyper Parameters 
        params = {
                'fs__k':list(range(60,126,5)),
                'clf__n_neighbors':list(range(1,11)),
                'clf__p':[1,2],
                'clf__weights':['uniform', 'distance']
                }

        # model fit
        results_knn = fit_classifier(SelectKBest(f_classif), KNeighborsClassifier(), params, X_selected, y_selected)

        # Knn CV score for comparision
        results = pd.DataFrame(results_knn.cv_results_['params'])
        results['test_score'] = results_knn.cv_results_['mean_test_score']
        results['metric'] = results['clf__p'].replace([1,2], ["Manhattan", "Euclidean"]) 

        # plot with different distance metrics
        plt.style.use("ggplot")
        for d in ["Manhattan", "Euclidean"]:
            temp = results[results['metric'] == d]
            temp_avg = temp.groupby('clf__n_neighbors').agg({'test_score': 'max'})
            plt.plot( temp_avg, marker= '.', label= d)    
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.ylabel("Mean CV Score")
        plt.title("KNN Performance Comparison")
        plt.show()

        # plot with different weight metric
        plt.style.use("ggplot")
        for w in ["uniform", "distance"]:
            temp = results[results['clf__weights'] == w]
            temp_avg = temp.groupby('clf__n_neighbors').agg({'test_score': 'max'})
            plt.plot( temp_avg , marker='.', label=w)    
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.ylabel("Mean CV Score")
        plt.title("KNN Performance Comparison")
        plt.show()

        # performance metrices for KNN classifier based on the best estimator
        # selected in Grid search
        robust_evaluation(results_knn.best_estimator_, X, y)

        # Learing curve KNN
        title = "Learning Curve - KNN"
        get_learning_curve(results_knn.best_estimator_, X, y, title)

    # decision tree
    elif model == DECISION_TREE : 

        print("\nRobust Evaluation--Decision Tree:\n")

        # Hyper Parameters 
        params = {
                'fs__k':list(range(65,126,5)),
                'clf__criterion':['gini', 'entropy'],
                'clf__max_depth':list(range(10,51,10)),
                'clf__min_samples_split':[2,3,4]
                }

        # model fit
        clf = DecisionTreeClassifier( random_state=111 )
        results_dt = fit_classifier(SelectKBest(f_classif), clf, params, X_selected, y_selected)

        # Decision tree CV score for comparision
        results = pd.DataFrame(results_dt.cv_results_['params'])
        results['test_score'] = results_dt.cv_results_['mean_test_score']

        # plot with different loss functions
        plt.style.use("ggplot")
        for c in ["gini", "entropy"]:
            temp = results[results['clf__criterion'] == c]
            temp_avg = temp.groupby('clf__max_depth').agg({'test_score': 'max'})
            plt.plot( temp_avg, marker= '.', label= c)
    
        plt.legend()
        plt.xlabel('Max Depth')
        plt.ylabel("Mean CV Score")
        plt.title("Decision Tree Performance Comparison")
        plt.show()

        # performance metrices for Descision Tree based on the best estimator
        # selected in Grid search
        robust_evaluation(results_dt.best_estimator_, X, y)

        # Learing curve Decision tree
        title = "Learning Curve - Decision Tree"
        get_learning_curve(results_dt.best_estimator_, X, y, title)

    # Bagging 
    elif model == BAGGING: 

        print("\nRobust Evaluation--Bagging:\n")

         # Hyper Parameters 
        params = {
                'fs__k':list(range(65,126,5)),
                'clf__n_estimators':[10, 50, 100, 150, 200, 250]
                } 

        # model fit
        clf = BaggingClassifier(random_state=111)
        results_bag = fit_classifier(SelectKBest(f_classif), clf, params, X_selected, y_selected)

        # Bagging CV score for comparision
        results = pd.DataFrame(results_bag.cv_results_['params'])
        results['test_score'] = results_bag.cv_results_['mean_test_score']
        plt.style.use("ggplot")
        temp_avg = results.groupby('clf__n_estimators').agg({'test_score': 'max'})
        plt.plot( temp_avg, marker= '.')
        plt.xlabel('Number of n estimators')
        plt.ylabel("Mean CV Score")
        plt.title("Bagging Performance Comparison")
        plt.show()

        # performance metrices for Bagging based on the best estimator
        # selected in Grid search
        robust_evaluation(results_bag.best_estimator_, X, y)

        # Learing curve Bagging
        title = "Learning Curve - Bagging"
        get_learning_curve(results_bag.best_estimator_, X, y, title)

    # random forest
    elif model == RANDOM_FOREST:

        print("\nRobust Evaluation--Random Forest:\n")

        # Hyper Parameters 
        params = {
                'fs__k':list(range(60,126,10)),
                'clf__n_estimators':[100, 200],
                'clf__criterion':['gini', 'entropy'],
                'clf__max_depth':list(range(20,51,10)),
                'clf__min_samples_split':[2,3]
                }

        # model fit
        clf = RandomForestClassifier(random_state=111)
        results_rf = fit_classifier(SelectKBest(f_classif), clf, params, X_selected, y_selected)

        # plot Random forest CV score for comparision
        results = pd.DataFrame(results_rf.cv_results_['params'])
        results['test_score'] = results_rf.cv_results_['mean_test_score']

        # plot for tree estimators
        plt.style.use("ggplot")
        for c in ["gini", "entropy"]:
            temp = results[results['clf__criterion'] == c]
            temp_avg = temp.groupby('clf__n_estimators').agg({'test_score': 'max'})
            plt.plot( temp_avg, marker= '.', label= c)
        plt.legend()
        plt.xlabel('Number of n estimators')
        plt.ylabel("Mean CV Score")
        plt.title("Random Forest Performance Comparison")
        plt.show()

        # plot for max depth
        plt.style.use("ggplot")
        for c in ["gini", "entropy"]:
            temp = results[results['clf__criterion'] == c]
            temp_avg = temp.groupby('clf__max_depth').agg({'test_score': 'max'})
            plt.plot( temp_avg, marker= '.', label= c)
        plt.legend()
        plt.xlabel('Max Depth')
        plt.ylabel("Mean CV Score")
        plt.title("Random Forest Performance Comparison")
        plt.show()

        # performance metrices for Random forest based on the best estimator
        # selected in Grid search
        robust_evaluation(results_rf.best_estimator_, X, y)

        # Learing curve Random forest
        title = "Learning Curve - Random Forest"
        get_learning_curve(results_rf.best_estimator_, X, y, title)

    # Svm
    elif model == SVM:

        print("\nRobust Evaluation--Support Vector Machine:\n")
        
        # Hyper Parameters 
        params = {
                'fs__k':list(range(60,126,10)),
                'clf__C':[0.01, 0.1, 1, 10, 50, 100],
                'clf__gamma':['auto', 'scale'],
                'clf__kernel':['rbf', 'poly', 'sigmoid'] # 
                }

        # model fit
        clf = SVC( random_state=111)
        results_svm = fit_classifier(SelectKBest(f_classif), clf, params, X_selected, y_selected)

        # plot SVM CV score for comparision
        results = pd.DataFrame(results_svm.cv_results_['params'])
        results['test_score'] = results_svm.cv_results_['mean_test_score']

        # plot for different penalty values
        plt.style.use("ggplot")
        for k in ['rbf', 'poly', 'sigmoid'] :
            temp = results[results['clf__kernel'] == k ]
            temp_avg = temp.groupby('clf__C').agg({'test_score': 'max'})
            plt.plot( temp_avg, marker= '.', label=k)    
        plt.legend()
        plt.xlabel('C - Penalty parameter')
        plt.ylabel("Mean CV Score")
        plt.title("SVM Performance Comparison")
        plt.show()

        # plot for gamma
        plt.style.use("ggplot")
        for k in ['rbf', 'poly', 'sigmoid'] :
            temp = results[results['clf__kernel'] == k]
            temp_avg = temp.groupby('clf__gamma').agg({'test_score': 'max'})
            plt.plot( temp_avg, marker= '.', label=k)    
        plt.legend()
        plt.xlabel("Gamma")
        plt.ylabel("Mean CV Score")
        plt.title("SVM Performance Comparison")
        plt.show()

        # performance metrices for SVM based on the best estimator
        # selected in Grid search
        robust_evaluation(results_svm.best_estimator_, X, y)

        # Learing curve SVM
        title = "Learning Curve - SVM"
        get_learning_curve(results_svm.best_estimator_, X, y, title)

def load_data(file_name):
    '''
    # Function that loads the dataset
    '''

    # from google.colab import files
    # uploaded = files.upload() 
    # import io
    # dataset = pd.read_csv(io.BytesIO(uploaded['file_name']))
 
    dataset = pd.read_csv(file_name, delimiter=",")
    
    return dataset


def main(model):
    ''' 
    Main function to evaluate classifers based on 'model'
    " Naive Bayes, Ridge Classifier, K-Nearest Neighbours, 
    Decision Tree, Bagging, Random Forest, Suport Vector Machines"
    '''

    # Load dataset
    dataset = load_data("dataset-sat.csv")

    if model == BASIC:
        # Run the baisc model
        print("\nBasic Models Results:")
        basic_model(dataset)

    elif model == CHECK:
        check_dataset(dataset)

    else:   
        # Run robust evaluation 
        robust_classification(dataset, model)


# Driver program

# pass the below constants in "main" function to get the resuts
# execute Models - BASIC / CHECK / NB / RIDGE/ KNN / DECISION_TREE / BAGGING / RANDOM_FOREST / SVM 
main(BASIC)