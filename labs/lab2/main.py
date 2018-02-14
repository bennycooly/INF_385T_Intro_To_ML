import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn import linear_model

from sklearn.preprocessing import PolynomialFeatures

# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn import metrics

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from scipy.stats import pearsonr

import pandas

import seaborn as sns

class ClassificationLab():
    def __init__(self):
        # feature names
        self.feature_names = None

        # dataset
        self.X = None
        self.y = None

        # split dataset
        self.X_train = None
        self.y_train = None

        self.X_test = None
        self.y_test = None

        # predicted data
        self.y_pred_tree = None
        self.y_pred_knn = None
        self.y_pred_svm = None
        self.y_pred_bayes = None

        self.skf = None

        # best params
        self.best_params_tree = {
            "criterion": "entropy",
            "max_depth": 5
        }

        self.best_params_knn = {
            "metric": "manhattan",
            "n_neighbors": 35
        }

        self.best_params_svm = {
            "C": 0.001,
            "degree": 2,
            "gamma": 0.0001
        }
    
    def load_dataset(self):
        df = pandas.read_excel("datasets/default of credit card clients.xls", header=1, index_col=0)

        # get feature names
        self.feature_names = df.columns
        # print(self.feature_names)

        # df_ndarr = df.sample(n=1000).values
        df_ndarr = df.values
        # split into x and y
        self.X, self.y = np.hsplit(df_ndarr, [23])
        
    
    def split_dataset(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
    
    def cross_validation_split(self, n_splits=2):
        self.skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    
    def train_decision_tree(self, **kwargs):
        tree = DecisionTreeClassifier(**kwargs).fit(self.X_train, self.y_train)
        self.y_pred_tree= tree.predict(self.X_test)
        return tree
    
    def test_hyperparams_tree(self):
        print("Testing hyperparameters for Decision Tree")
        tuned_parameters = {
            "criterion": ["gini", "entropy"],
            "max_depth": range(1, 11)
        }

        clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, scoring="accuracy", cv=self.skf)

        clf.fit(self.X_train, self.y_train.ravel())
        print("Best accuracy: ", clf.best_score_)
        print("Best params: ", clf.best_params_)

        self.best_params_tree = clf.best_params_
    
    def test_hyperparams_knn(self):
        print("Testing hyperparameters for K-Nearest Neighbors")
        tuned_parameters = {
            "n_neighbors": range(1, 101),
            "metric": ["manhattan", "euclidean"]
        }

        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, scoring="accuracy", cv=self.skf)

        clf.fit(self.X_train, self.y_train.ravel())
        print("Best accuracy: ", clf.best_score_)
        print("Best params: ", clf.best_params_)

        self.best_params_knn = clf.best_params_
    
    def test_hyperparams_svm(self):
        print("Testing hyperparameters for SVM")
        tuned_parameters = {
            "C": np.logspace(-3, 2, 5),
            "degree": range(2, 5),
            "gamma": np.logspace(-4, -2, 5)
        }

        clf = GridSearchCV(SVC(), tuned_parameters, scoring="accuracy", cv=self.skf)

        clf.fit(self.X_train, self.y_train.ravel())
        print("Best accuracy: ", clf.best_score_)
        print("Best params: ", clf.best_params_)

        self.best_params_svm = clf.best_params_
    

        
    def evaluate_models(self):
        plt.figure()

        # Decision Tree
        
        print("Decision Tree Metrics")
        tree = DecisionTreeClassifier(**self.best_params_tree).fit(self.X_train, self.y_train.ravel())
        self.y_pred_tree = tree.predict(self.X_test)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred_tree))
        print(metrics.classification_report(self.y_test, self.y_pred_tree))
        
        # confusion matrix
        plt.subplot(221)
        mat = metrics.confusion_matrix(self.y_test, self.y_pred_tree)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.title("Decision Tree")
        plt.xlabel("True label")
        plt.ylabel("Predicted label")


        # KNN

        print("K-Nearest Neighbors Metrics")
        knn = KNeighborsClassifier(**self.best_params_knn).fit(self.X_train, self.y_train.ravel())
        self.y_pred_knn = knn.predict(self.X_test)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred_knn))
        print(metrics.classification_report(self.y_test, self.y_pred_knn))

        # confusion matrix
        plt.subplot(222)
        mat = metrics.confusion_matrix(self.y_test, self.y_pred_knn)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.title("KNN")
        plt.xlabel("True label")
        plt.ylabel("Predicted label")


        # SVM

        print("SVM Metrics")
        svm = SVC(**self.best_params_svm).fit(self.X_train, self.y_train.ravel())
        self.y_pred_svm = svm.predict(self.X_test)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred_svm))
        print(metrics.classification_report(self.y_test, self.y_pred_svm))

        # confusion matrix
        plt.subplot(223)
        mat = metrics.confusion_matrix(self.y_test, self.y_pred_svm)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.title("SVM")
        plt.xlabel("True label")
        plt.ylabel("Predicted label")


        # Naive Bayes

        print("Naive Bayes Metrics")
        gaussian = GaussianNB().fit(self.X_train, self.y_train.ravel())
        self.y_pred_bayes = gaussian.predict(self.X_test)
        print("Accuracy: ", metrics.accuracy_score(self.y_test, self.y_pred_bayes))
        print(metrics.classification_report(self.y_test, self.y_pred_bayes))

        # confusion matrix
        plt.subplot(224)
        mat = metrics.confusion_matrix(self.y_test, self.y_pred_bayes)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
        plt.title("Naive Bayes")
        plt.xlabel("True label")
        plt.ylabel("Predicted label")

        plt.tight_layout()
        plt.savefig("confusion-matrices.png")
        plt.show()




def q1():
    lab = ClassificationLab()
    lab.load_dataset()
    lab.split_dataset()
    print("X_train: ", lab.X_train)
    print("X_test: ", lab.X_test)
    print("y_train: ", lab.y_train)
    print("y_test: ", lab.y_test)

def q2():
    lab = ClassificationLab()
    lab.load_dataset()
    lab.split_dataset()
    lab.cross_validation_split()
    lab.test_hyperparams_tree()
    lab.test_hyperparams_knn()
    lab.test_hyperparams_svm()


def q3():
    lab = ClassificationLab()
    lab.load_dataset()
    lab.split_dataset()
    lab.evaluate_models()

def main():
    print("Running question 1")
    q1()
    print("\n")

    print("Running question 2")
    q2()
    print("\n")

    print("Running question 3")
    q3()

if __name__ == "__main__":
    main()
