# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Initialize model**
   Define ( y = w_0 + w_1x_1 + \dots + w_nx_n ) and set initial weights.

2. **Apply SGD update**
   Update weights for each sample:
   ( w_j = w_j - \alpha (y - \hat{y})x_j )

3. **Train iteratively**
   Loop through data one sample at a time for multiple epochs.

4. **Check convergence**
   Stop when error (e.g., MSE) becomes minimal or stable

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Ranjani S
RegisterNumber: 212225230224
import numpy as np

class MultivariateLinearRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        
        self.weights = np.zeros(n_features)
        self.bias = 0

       
        for _ in range(self.epochs):
            for i in range(n_samples):
                y_pred = np.dot(X[i], self.weights) + self.bias

                
                dw = -2 * X[i] * (y[i] - y_pred)
                db = -2 * (y[i] - y_pred)
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias



if __name__ == "__main__":
    
    X = np.array([
        [1, 2],
        [2, 3],
        [4, 5],
        [3, 6]
    ])
    
    y = np.array([5, 8, 14, 13])

    model = MultivariateLinearRegressionSGD(learning_rate=0.01, epochs=1000)
    model.fit(X, y)

    predictions = model.predict(X)

    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Predictions:", predictions)



*/
```

## Output:

<img width="954" height="808" alt="Screenshot 2026-04-29 104914" src="https://github.com/user-attachments/assets/51967c8f-6338-4aae-b7e7-3de732d3ecb0" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
