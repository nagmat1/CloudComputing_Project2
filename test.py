import pandas as pd
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


# function to get cross validation scores
def get_cv_scores(model):
    scores = cross_val_score(model,
                             x_train,
                             y_train,
                             cv=5,
                             scoring='r2')

    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')


np.random.seed(0)
print("Loading dataset...")
df = pd.read_csv('winequality-white.csv', encoding='utf8', chunksize=None,delimiter=";")

# Define features and target
features = list(df.columns)
features.remove('quality')

x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['quality'], test_size=0.2, shuffle=True, stratify=df['quality'])

# Linear Regression
regression_model = LinearRegression()
regression_model.fit(x_train,y_train)
#predict
y_predicted = regression_model.predict(x_test)

mse = cross_val_score(regression_model, x_train, y_train, scoring="neg_mean_squared_error", cv=5)
mean_mse = np.mean(mse)
print("Mean_mse = ",mean_mse)

scores = cross_val_score(regression_model, x_train,y_train,cv=5,scoring='r2')
print("CV mean = ",np.mean(scores))
print("STD : {} \n".format(np.std(scores)))

#model evaluation
rmse = mean_squared_error(y_test,y_predicted)
r2 = r2_score(y_test,y_predicted)

# printing values
print('Slope:' ,regression_model.coef_)
print('Intercept:', regression_model.intercept_)
print('Root mean squared error: ', rmse)
print('R2 score: ', r2)

# Train model
lr = LinearRegression().fit(x_train,y_train)
get_cv_scores(lr)

ridge = Ridge(alpha=1).fit(x_train,y_train)
get_cv_scores(ridge)

lasso = Lasso(alpha=1).fit(x_train,y_train)
get_cv_scores(lasso)


alpha = [0.001,0.01, 0.1, 1, 10, 100,1000]
param_grid = dict(alpha=alpha)

grid = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='r2',verbose=1, n_jobs=-1)
grid_result = grid.fit(x_train,y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

from sklearn.ensemble import RandomForestRegressor
labels = np.array(['quality'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = model_selection.train_test_split(df[features], df['quality'], test_size=0.2, shuffle=True)
rf = RandomForestRegressor(n_estimators=1000,random_state=42)
rf.fit(x_train,y_train)
predictions = rf.predict(x_test)
errors = abs(predictions - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

mape = 100 * (errors / y_test )
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

import graphviz as gv
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth = 3)
rf_small.fit(x_train, y_train)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
