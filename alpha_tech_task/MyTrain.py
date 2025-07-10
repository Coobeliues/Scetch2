import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LassoCV, Ridge
from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import cross_val_score

import MyModels


# rmse cross validationz
def rmse_cv(model, X_train, y):
    rmse = np.sqrt(
        -cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=10)
    )

    return rmse

# с лучшими параметрами 
def create_tuned_models(X_train, y):
    best_params_lgbm = MyModels.tune_lgbm(X_train, y)
    best_params_xgb = MyModels.tune_xgb(X_train, y)
    best_params_rf = MyModels.tune_rf(X_train, y)
    best_params_gb = MyModels.tune_gb(X_train, y)
    
    # best_params_knn = MyModels.tune_knn(X_train, y)
    # best_params_svr = MyModels.tune_svr(X_train, y)
    # best_params_ada = MyModels.tune_ada(X_train, y)
    # best_params_ridge = MyModels.tune_ridge(X_train, y)
    # best_params_lasso = MyModels.tune_lasso(X_train, y)
    # best_params_enet = MyModels.tune_enet(X_train, y)

    models_tuned = {
        # "Ridge": Ridge(alpha=best_params_ridge),
        "Random Forest": RandomForestRegressor(**best_params_rf, random_state=42, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(**best_params_gb, random_state=42),
        "XGB": XGBRegressor(**best_params_xgb, random_state=42, n_jobs=1),
        # "LassoCV": LassoCV(**best_params_lasso, n_jobs=1),
        "LGBMRegressor": LGBMRegressor(**best_params_lgbm, random_state=42, verbose=-1, n_jobs=1),
        # "LinearRegression": LinearRegression(n_jobs=1),
    }

    return models_tuned


# def trainEvaluate_models(models_tuned, X_train, y, flag=None):
#     for model_name, model in models_tuned.items():
#         score = rmse_cv(model, X_train, y).mean()
#         print(f"{model_name} RMSE: {score}")

#         if flag is not None:
#             if flag == True:
#                 model.fit(X_train, y)


# def rmse_cv(model, X, y):
#     rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=10))
#     return rmse


def train_and_evaluate(models, X, y):
    for name, model in models.items():
        score = rmse_cv(model, X, y).mean()

        model.fit(X, y)
        y_pred = model.predict(X)

        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        print(f"{name}:")
        print(f"  RMSE = {score:.4f}")
        print(f"  MAE  = {mae:.4f}")
        print(f"  R²   = {r2:.4f}")
        print("-" * 30)

# сохранение результатов в Csv
# def savePred(models_tuned, test, X_test):
#     for model_name, model in models_tuned.items():
#         predictions = pd.DataFrame(
#             {"Id": test["Id"], "SalePrice": np.expm1(model.predict(X_test))}
#         )
#         # # Model Information
#         # model_params = str(model.get_params())  # Convert model parameters to string
#         # preproc_methods = str(X_train.columns)  # Just an example, modify as per your preprocessing method
#         # score_rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)).mean()

#         # model_info = pd.DataFrame({
#         #     "ModelName": [model_name],
#         #     "ModelParams": [model_params],
#         #     "PreprocessingMethods": [preproc_methods],
#         #     "ScoreRMSE": [score_rmse],
#         # })

#         # # Append model information to the results list
#         # results.append(model_info)
#         predictions.to_csv(
#             f"predictions/predictions_{model_name}.csv",
#             index=False,
#         )

# графики остатков и сохр
def plot_and_save_residuals(models, X_train, X_test, y_train, y_test, save_dir='residuals', inverse_log=True):
    os.makedirs(save_dir, exist_ok=True)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # обратно из log1p, если нужно
        if inverse_log:
            y_train_pred = np.expm1(y_train_pred)
            y_test_pred = np.expm1(y_test_pred)
            y_train_true = np.expm1(y_train)
            y_test_true = np.expm1(y_test)
        else:
            y_train_true = y_train
            y_test_true = y_test

        plt.figure(figsize=(10, 6))

        plt.scatter(y_train_pred, y_train_pred - y_train_true,
                    c='black', marker='o', s=35, alpha=0.5, label='Train data')

        plt.scatter(y_test_pred, y_test_pred - y_test_true,
                    c='cyan', marker='o', s=35, alpha=0.7, label='Test data')

        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.xlabel("Predicted charges (USD)")
        plt.ylabel("Residuals (Prediction - Actual)")
        plt.title(f"Residual plot: {name}")
        plt.legend(loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"Residual_{name.replace(' ', '_')}.png")
        plt.savefig(save_path)
        plt.close()