
import os
import json

import random

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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ensemble
from sklearn.ensemble import VotingClassifier

# metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# datasets
from sklearn.datasets import fetch_mldata

from sklearn.decomposition import PCA

# local imports
import vizwiz_api.vqaEvaluation

# Azure API
from azure_api.computer_vision import VisionApi
from azure_api.text import TextApi

# data directory
data_dir = "data/"
annotations_dir = data_dir + "Annotations/"
images_dir = data_dir + "Images/"

results_dir = "results/"

# subscription key for Azure
cv_subscription_key = "62ad92e01c484a74a3041b11d41d4556"
text_subscription_key = "6849d779257149c7acbfe524328670fd"

class VizWizLab():
    def __init__(self):
        self.data = ""
        self.dataset_type = ""
        self.classifier = None
        self.classifier_name = ""
        self.components = []

        self.annotations = {
            "train": None,
            "val": None,
            "test": None
        }

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

        # API's
        self.vision_api = VisionApi(cv_subscription_key)
        self.text_api = TextApi(text_subscription_key)
    
    def load_dataset(self):
        print("Loading dataset...")
        
        with open(annotations_dir + "train.json", 'r') as f_train, \
                open(annotations_dir + "val.json", 'r') as f_val, \
                open(annotations_dir + "test.json", 'r') as f_test:
            self.annotations["train"] = json.load(f_train)
            self.annotations["val"] = json.load(f_val)
            self.annotations["test"] = json.load(f_test)
        
        self.annotations["train"] = random.choices(self.annotations.get("train"), k=1000)
        self.annotations["val"] = random.choices(self.annotations.get("val"), k=300)

        # Test is the first 30 annotations
        self.annotations["test"] = self.annotations.get("test")[:30]

        self.y_train = np.array(list(map(lambda a: a.get("answerable"), self.annotations.get("train"))))
        self.y_val = np.array(list(map(lambda a: a.get("answerable"), self.annotations.get("val"))))
        print(self.y_train)

        # extract features
        self.X_train = self.extract_features(self.annotations["train"])
        self.X_val = self.extract_features(self.annotations["val"])
        self.X_test = self.extract_features(self.annotations["test"])

        print(self.X_train)
        
    
    def extract_features(self, annotations):
        print("Extracting features...")

        all_features = []

        # print(annotations)

        for i, annotation in enumerate(annotations):
            print("extracting feature for " + str(i))
            features = []

            image_file = images_dir + annotation.get("image")
            res = self.vision_api.analyze_image(image_file)

            features.extend(res)

            # question = annotation.get("question")
            # question_features = self.text_api.analyze_text(question)

            # features.extend(question_features)

            all_features.append(features)
        
        questions = list(map(lambda a: a.get("question"), annotations))
        all_questions_features = self.text_api.analyze_text(questions)
        for i, question_features in enumerate(all_questions_features):
            all_features[i].extend(question_features)
        
        return np.array(all_features)


    
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
        print("Training classifier...")
        clf1 = DecisionTreeClassifier()
        clf2 = SVC()
        clf3 = LogisticRegression()

        eclf = VotingClassifier(estimators=[("tree", clf1), ("svm", clf2), ("lr", clf3)])

        eclf.fit(self.X_train, self.y_train)
        accuracy = eclf.score(self.X_val, self.y_val)

        print("Accuracy: " + str(accuracy))

        self.y_test = eclf.predict(self.X_test)
        json_results = []
        for i, answer in enumerate(self.y_test):
            json_results.append({
                "answerable": np.asscalar(answer),
                "image": self.annotations.get("test")[i].get("image")
            })
        
        # Write results to file
        # print(json_results)
        results_file = results_dir + "predictions.json"
        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=4)



def main():
    lab = VizWizLab()
    lab.load_dataset()
    # lab.split_dataset()
    # lab.standardize_dataset()
    lab.train_classifier()

if __name__ == "__main__":
    main()
