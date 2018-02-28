
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

# classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# ensemble
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

# metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# datasets
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

from sklearn.decomposition import PCA


class PCALab():
    def __init__(self):
        self.data = ""
        self.dataset_type = ""
        self.classifier = None
        self.classifier_name = ""
        self.components = []

        self.X = None
        self.X_train = None
        self.X_train_reduced_vals = []
        self.X_test = None
        self.X_test_reduced_vals = []

        self.y = None
        self.y_train = None
        self.y_test = None
        self.y_pred_vals = None

        self.num_components_vals = []

        self.accuracy_scores = []
    
    def load_dataset(self, type):
        print("Loading dataset...")
        self.dataset_type = type
        if type == "MNIST":
            mnist = fetch_mldata("MNIST original", data_home="datasets/")
            idx = np.random.randint(len(mnist.data), size=10000)
            self.X = mnist.data[idx]
            self.y = mnist.target[idx]
        
        elif type == "LFW":
            lfw = fetch_lfw_people(min_faces_per_person=60, data_home="datasets/")
            self.X = lfw.data
            self.y = lfw.target
        
        elif type == "Cancer":
            cancer = load_breast_cancer()
            self.X = cancer.data
            self.y = cancer.target
        
        elif type == "Iris":
            iris = load_iris()
            self.X = iris.data
            self.y = iris.target
    
    def split_dataset(self, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
        
    def transform_dataset(self):
        print("Transforming dataset...")
        self.num_components_vals = np.linspace(1, len(self.X_train[0]), num=20, dtype=np.dtype(np.int64))
        self.X_train_reduced_vals = []
        self.X_test_reduced_vals = []
        print(self.X_train.shape)
        for num_components in self.num_components_vals:
            print("num components:", num_components)
            pca = PCA(n_components=num_components).fit(self.X_train)
            self.X_train_reduced_vals.append(pca.transform(self.X_train))
            self.X_test_reduced_vals.append(pca.transform(self.X_test))
            
        print(self.X_train_reduced_vals[2].shape)
        print(self.X_test_reduced_vals[2].shape)
        print(self.num_components_vals)
    
    def train_classifier(self, classifier):
        self.classifier_name = classifier
        if classifier == "DecisionTree":
            self.classifier = DecisionTreeClassifier()
        
        elif classifier == "KNN":
            self.classifier = KNeighborsClassifier(n_neighbors=100)
        
        elif classifier == "SVM":
            self.classifier = SVC()
        
        elif classifier == "NaiveBayes":
            self.classifier = GaussianNB()

        self.y_pred_vals = []
        self.accuracy_scores = []
        # Loop over all reduced X training sets
        for i, X_train_reduced in enumerate(self.X_train_reduced_vals):
            self.classifier.fit(X_train_reduced, self.y_train)
            self.y_pred_vals.append(self.classifier.predict(self.X_test_reduced_vals[i]))

        for y_pred in self.y_pred_vals:
            # print("Pred:", y_pred)
            # print("Actual:", self.y_test)
            self.accuracy_scores.append(accuracy_score(self.y_test, y_pred))
        
        print(self.num_components_vals)
        print(self.accuracy_scores)
    
    def plot_statistics(self):
        plt.figure()
        plt.title("Accuracy vs Components: " + self.dataset_type + ", " + self.classifier_name)
        plt.plot(self.num_components_vals, self.accuracy_scores)
        plt.xlabel("Number of PCA Components")
        plt.ylabel("Accuracy")

        plt.savefig(os.path.join("plots", "accuracy_vs_components_" + self.dataset_type + "_" + self.classifier_name))
        # plt.show()


    def train_normal(self):
        clf = KNeighborsClassifier().fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        print(accuracy_score(self.y_test, y_pred))


class EnsembleLab():
    def __init__(self):
        self.dataset_type = ""
        self.classifier = None
        self.classifier_name = ""
        self.components = []

        self.X = None
        self.X_train = None
        self.X_train_reduced_vals = []
        self.X_test = None
        self.X_test_reduced_vals = []

        self.y = None
        self.y_train = None
        self.y_test = None
        self.y_pred_vals = None

        self.num_components_vals = []

        self.accuracy_scores = []
    
    def load_dataset(self, type):
        print("Loading dataset...")
        self.dataset_type = type
        
        if type == "Cancer":
            cancer = load_breast_cancer()
            self.X = cancer.data
            self.y = cancer.target
        
        elif type == "Iris":
            iris = load_iris()
            self.X = iris.data
            self.y = iris.target
    
    def split_dataset(self, test_size=0.3):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
    
    def train(self):
        clf1 = KNeighborsClassifier()
        clf1_scores = cross_val_score(clf1, self.X, self.y, cv=10)
        print("Mean accuracy (KNN): " + str(clf1_scores.mean()))

        clf2 = GaussianNB()
        clf2_scores = cross_val_score(clf2, self.X, self.y, cv=10)
        print("Mean accuracy (Naive Bayes): " + str(clf2_scores.mean()))

        clf3 = DecisionTreeClassifier()
        clf3_scores = cross_val_score(clf3, self.X, self.y, cv=10)
        print("Mean accuracy (Decision Tree): " + str(clf3_scores.mean()))

        eclf = VotingClassifier(estimators=[("knn", clf1), ("gnb", clf2), ("dt", clf3)], voting="hard")
        ensemble_scores = cross_val_score(eclf, self.X, self.y, cv=10)
        print("Mean accuracy (Ensemble): " + str(ensemble_scores.mean()))

        bagging = BaggingClassifier(clf1)
        bagging_scores = cross_val_score(bagging, self.X, self.y, cv=10)
        print("Mean accuracy (Bagging, KNN): " + str(bagging_scores.mean()))

        adabooster = AdaBoostClassifier(n_estimators=100)
        adabooster_scores = cross_val_score(adabooster, self.X, self.y, cv=10)
        print("Mean accuracy (Adabooster, Decision Tree): " + str(adabooster_scores.mean()))




def q1():
    lab = PCALab()
    # Dataset: LFW
    # lab.load_dataset("LFW")
    # lab.split_dataset()
    # lab.transform_dataset()

    # lab.train_classifier("DecisionTree")
    # lab.plot_statistics()

    # lab.train_classifier("KNN")
    # lab.plot_statistics()

    # Dataset: MNIST
    lab.load_dataset("MNIST")
    lab.split_dataset()
    lab.transform_dataset()

    lab.train_classifier("DecisionTree")
    lab.plot_statistics()

    lab.train_classifier("KNN")
    lab.plot_statistics()
    # lab.train_normal()
    

def q2():
    lab = EnsembleLab()
    lab.load_dataset("Cancer")
    lab.train()


def main():
    print("Running question 1")
    q1()
    print("\n")

    print("Running question 2")
    q2()

if __name__ == "__main__":
    main()
