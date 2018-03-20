
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler

# classifiers
from sklearn.neural_network import MLPClassifier

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


class MLPLab():
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
    
    def load_dataset(self):
        print("Loading dataset...")
        mnist = fetch_mldata("MNIST original", data_home="datasets/")
        idx = np.random.randint(len(mnist.data), size=10000)
        self.X = mnist.data[idx]
        self.y = mnist.target[idx]
    
    def split_dataset(self, test_size=0.3):
        print("Splitting dataset...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
    
    def standardize_dataset(self):
        print("Standardizing dataset...")
        stdsc = StandardScaler()
        stdsc.fit(self.X_train)
        self.X_train = stdsc.transform(self.X_train)
        self.X_test = stdsc.transform(self.X_test)
        
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
    
    def train_classifier(self):
        params = {
            "hidden_layer_sizes": [(100), (100, 100), (100, 100, 100)]
        }
        clf = GridSearchCV(estimator=MLPClassifier(), param_grid=params)
        clf.fit(self.X_train, self.y_train)
        score = clf.score(self.X_test, self.y_test)
        print(clf.best_params_)
        print("Accuracy: " + str(score))


def main():
    lab = MLPLab()
    lab.load_dataset()
    lab.split_dataset()
    lab.standardize_dataset()
    lab.train_classifier()

if __name__ == "__main__":
    main()
