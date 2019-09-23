# simple impute
# local score 18125
# kaggle score 16640
# minimize score

import pandas as pd
import sys  # noqa
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# Read the data
X_full = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))


# imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

final_X_train = imputed_X_train
final_X_valid = imputed_X_valid

my_model = RandomForestRegressor(n_estimators=100, random_state=0)

# Fit the model to the training data & validate for score
my_model.fit(final_X_train, y_train)
preds_valid = my_model.predict(final_X_valid)
score = mean_absolute_error(y_valid, preds_valid)
print(f'score: {score}')

final_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))

preds_test = my_model.predict(final_X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
