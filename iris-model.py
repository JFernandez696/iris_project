#Import main libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import pickle

#Load data

iris = datasets.load_iris()

X = iris.data
y = iris.target

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y)

lin_reg = LinearRegression()
log_reg = LogisticRegression()
svm_reg = SVC()

#Fit the models

lin_regr = lin_reg.fit(X_train, y_train)
log_regr = log_reg.fit(X_train, y_train)
svm_regr = svm_reg.fit(X_train, y_train)

with open('lin_reg.pkl', 'wb') as li:
    pickle.dump(lin_regr, li)

with open('log_reg.pkl', 'wb') as lo:
    pickle.dump(log_regr, lo)

with open('svm_reg.pkl', 'wb') as sv:
    pickle.dump(svm_regr, sv)