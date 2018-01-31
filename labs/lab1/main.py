import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

import pandas

class RegressionLab():
    def __init__(self):
        # dataset
        self.X = None
        self.X_poly = None
        self.y = None

        # split dataset
        self.X_train = None
        self.X_poly_train = None
        self.y_train = None
        self.y_poly_train = None

        self.X_test = None
        self.X_poly_test = None
        self.y_test = None
        self.y_poly_test = None

        # predicted data
        self.y_pred_lr = None
        self.y_pred_ridge = None
        self.y_pred_lasso = None
        self.y_pred_poly = None



    def gen_quadratic_dataset(self, num_samples=1000, a=1, b=2, c=1, x_max=4):
        self.X = x_max * np.random.rand(num_samples, 1)
        self.y = (a * self.X * self.X) + (b * self.X) + c + np.random.randn(num_samples, 1)
    
    def load_wine_dataset(self):
        df = pandas.read_csv("datasets/winequality-red.csv", sep=';')
        df_ndarr = df.values
        # split into x and y
        self.X, self.y = np.hsplit(df_ndarr, [11])
    
    def split_dataset(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
    
    def linear_regression(self):
        regr = linear_model.LinearRegression()
        regr.fit(self.X_train, self.y_train)
        self.y_pred_lr = regr.predict(self.X_test)

    
    def ridge_regression(self, alpha=1.0):
        regr = linear_model.Ridge(alpha=alpha)
        regr.fit(self.X_train, self.y_train)
        self.y_pred_ridge = regr.predict(self.X_test)
    
    def lasso_regression(self, alpha=1.0):
        regr = linear_model.Lasso(alpha=alpha)
        regr.fit(self.X_train, self.y_train)
        self.y_pred_lasso = regr.predict(self.X_test)

    
    def poly_regression(self):
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        # reuse the previous dataset
        self.X_poly_train = poly_features.fit_transform(self.X_train)
        self.X_poly_test = poly_features.fit_transform(self.X_test)
        regr = linear_model.LinearRegression()
        regr.fit(self.X_poly_train, self.y_train)
        self.y_pred_poly = regr.predict(self.X_poly_test)
        
    
    def save_plots(self):
        plt.figure()
        plt.plot(self.X, self.y, "b.")
        plt.savefig("q1/dataset.png")
        plt.clf()
        plt.close()
        
        # linear
        plt.figure()
        plt.plot(self.X_test, self.y_test, "b.")
        plt.plot(self.X_test, self.y_pred_lr, "r-")
        plt.savefig("q1/linear_regression.png")
        plt.clf()
        plt.close()

        # ridge
        plt.figure()
        plt.plot(self.X_test, self.y_test, "b.")
        plt.plot(self.X_test, self.y_pred_ridge, "r-")
        plt.savefig("q1/ridge_regression.png")
        plt.clf()
        plt.close()

        # lasso
        plt.figure()
        plt.plot(self.X_test, self.y_test, "b.")
        plt.plot(self.X_test, self.y_pred_lasso, "r-")
        plt.savefig("q1/lasso_regression.png")
        plt.clf()
        plt.close()

        # polynomial
        # plot the x feature
        plt.figure()
        X_poly_test_split = np.array(np.hsplit(self.X_poly_test, 2)[0])
        print(X_poly_test_split.shape)
        plt.plot(X_poly_test_split, self.y_test, "b.")
        plt.plot(X_poly_test_split, self.y_pred_poly, "r.")
        plt.savefig("q1/poly_regression.png")
        plt.clf()
        plt.close()

    def print_statistics(self):
        # calculate error
        print("MAE for linear regression: ", mean_absolute_error(self.y_test, self.y_pred_lr))
        print("Correlation Coefficient for linear regression: ", pearsonr(self.y_test, self.y_pred_lr)[0])

        print("MAE for ridge regression: ", mean_absolute_error(self.y_test, self.y_pred_ridge))
        print("Correlation Coefficient for ridge regression: ", pearsonr(self.y_test, self.y_pred_ridge)[0])

        print("MAE for lasso regression: ", mean_absolute_error(self.y_test, self.y_pred_lasso))
        print("Correlation Coefficient for lasso regression: ", pearsonr(self.y_test, np.reshape(self.y_pred_lasso, len(self.y_pred_lasso), 1))[0])

        print("MAE for polynomial regression: ", mean_absolute_error(self.y_test, self.y_pred_poly))
        print("Correlation Coefficient for polynomial regression: ", pearsonr(self.y_test, self.y_pred_poly)[0])



def q1():
    lab = RegressionLab()
    lab.gen_quadratic_dataset()
    lab.split_dataset()
    lab.linear_regression()
    lab.ridge_regression()
    lab.lasso_regression()
    lab.poly_regression()

    lab.save_plots()
    lab.print_statistics()

def q2():
    lab = RegressionLab()
    lab.load_wine_dataset()
    lab.split_dataset()
    lab.linear_regression()
    lab.ridge_regression()
    lab.lasso_regression()
    lab.poly_regression()

    lab.print_statistics()

def main():
    print("Running question 1")
    q1()

    print("\n")

    print("Running question 2")
    q2()

if __name__ == "__main__":
    main()
