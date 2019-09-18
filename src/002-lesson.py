# simple impute
# local score 23528
# kaggle score 21768 - SHOULD BE 16640.85750
# minimize score

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))


# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

my_model = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)

# Fit the model to the training data & validate for score
my_model.fit(imputed_X_train, y_train)
preds_valid = my_model.predict(imputed_X_valid)
score = mean_absolute_error(y_valid, preds_valid)
print(f'score: {score}')

# Generate test predictions on full set
my_model.fit(imputed_X_train, y_train)

imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

preds_test = my_model.predict(imputed_X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
