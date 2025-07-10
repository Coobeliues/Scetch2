from lightgbm import LGBMRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures


# LGBMRegressor
def tune_lgbm(X_train, y):
    lgbm_params = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7], 'num_leaves': [7, 15, 31, 63]}
    grid_lgbm = GridSearchCV(LGBMRegressor(random_state=42, verbose=-1), param_grid=lgbm_params, n_jobs=1)
    grid_lgbm.fit(X_train, y)

    return grid_lgbm.best_params_


# XGBRegressor
def tune_xgb(X_train, y):
    xgb_params = {'learning_rate': [0.01, 0.05, 0.1], 'n_estimators': [100, 300, 500], 'max_depth': [3, 4, 5], 'subsample': [0.8, 0.9, 1.0], 'colsample_bytree': [0.8, 0.9, 1.0]}
    grid_xgb = GridSearchCV(XGBRegressor(random_state=42), param_grid=xgb_params, n_jobs=1)
    grid_xgb.fit(X_train, y)

    return grid_xgb.best_params_


# RandomForestRegressor
def tune_rf(X_train, y):
    param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [5, 10], 'min_samples_leaf': [1, 2, 4]}
    grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid=param_grid_rf, n_jobs=1)
    grid_rf.fit(X_train, y)

    return grid_rf.best_params_


# GradientBoostingRegressor
def tune_gb(X_train, y):
    param_grid_gb = {'n_estimators': [200, 300], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4], 'subsample': [0.8, 0.9, 1.0]}
    grid_gb = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid=param_grid_gb, n_jobs=1)
    grid_gb.fit(X_train, y)

    return grid_gb.best_params_



