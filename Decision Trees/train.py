from sklearn import datasets
from DecisionTress import DTR
from sklearn.model_selection import train_test_split
import numpy as np

data = datasets.load_breast_cancer()
x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

clf = DTR(max_depth=10)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
acc = np.sum(y_test==y_pred)/len(y_test)

print(acc)

