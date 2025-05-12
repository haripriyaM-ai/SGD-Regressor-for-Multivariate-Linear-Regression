# EXPERIMENT NO: 02
# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries: NumPy, datasets, model classes, preprocessing tools, and evaluation metrics from sklearn.

2. Load the California Housing dataset using fetch_california_housing().

3. Select the first 3 features as input (features) and combine the target and the 7th feature as output (targets).

4. Split the features and targets into training and testing sets using train_test_split() with test_size=0.2 and random_state=42.

5. Initialize StandardScaler for both input (scaler_input) and output (scaler_output) data.

6. Apply scaling:

     i.Fit and transform X_train using scaler_input, and transform X_test.

     ii.Fit and transform y_train using scaler_output, and transform y_test.

7. Create a base model using SGDRegressor(max_iter=1000, tol=1e-3).

8. Wrap the base model with MultiOutputRegressor to handle multiple outputs.

9. Train the model on the scaled training data using .fit().

10. Predict the outputs for the scaled test data using .predict().

11. Inverse transform the predicted and actual test outputs using scaler_output to return them to their original scale.

12. Calculate the Mean Squared Error (MSE) between actual and predicted outputs using mean_squared_error().

13. Print the MSE and display the first 5 predicted output rows for sample verification.
 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: HARI PRIYA M
RegisterNumber: 212224240047 
*/
```

    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.linear_model import SGDRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    
    housing_data = fetch_california_housing()
    features = housing_data.data[:, :3]
    targets = np.column_stack((housing_data.target, housing_data.data[:, 6]))
    
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    scaler_input = StandardScaler()
    scaler_output = StandardScaler()
    
    X_train = scaler_input.fit_transform(X_train)
    X_test = scaler_input.transform(X_test)
    
    y_train = scaler_output.fit_transform(y_train)
    y_test = scaler_output.transform(y_test)
    
    base_model = SGDRegressor(max_iter=1000, tol=1e-3)
    multi_output_model = MultiOutputRegressor(base_model)
    multi_output_model.fit(X_train, y_train)
    
    y_pred = multi_output_model.predict(X_test)
    
    y_pred = scaler_output.inverse_transform(y_pred)
    y_test = scaler_output.inverse_transform(y_test)
    
    mse_score = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error on test data:", mse_score)
    
    print("\nPredicted outputs(first five rows):\n", y_pred[:5])

## Output:
![Screenshot 2025-05-12 220509](https://github.com/user-attachments/assets/6730c89d-4c33-4f35-a242-997a67213fb6)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
