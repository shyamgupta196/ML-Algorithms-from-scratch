import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from lr import LR


X,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

fig = plt.figure(figsize=(8,6)) 
plt.scatter(x[:0],y,color="b",s=10)
plt.show()

reg = LR(lr = 0.01)
reg.fit(X_train,y_train)
preds = reg.predict(X_test) 

def MSE(y_test,preds):
    return np.mean((y_test-preds)**2)

mse = MSE(y_test,preds)
print(mse)

y_pred_line = reg.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train,y_train,color=cmap(0.9),s=10)
m2 = plt.scatter(X_test,y_test,color=cmap(0.5),s=10)
plt.plot(X,y_pred_line,color='black',linewidth=2,label="Prediction")
plt.show()
