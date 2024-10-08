
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


df = pd.read_csv('housing.csv')


X = df[['number_of_bedrooms', 'number_of_bathrooms', 'square_footage']]
y = df['price']


pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler()),                 
    ('model', LinearRegression())                  
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation Mean Squared Error (MSE): {-cross_val_scores.mean()}")


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Test Mean Squared Error: {mse}")


results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print("\nPredictions vs Actual Values:")
print(results)


import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # 45-degree line
plt.show()
