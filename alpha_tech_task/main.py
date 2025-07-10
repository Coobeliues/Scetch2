import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.model_selection import train_test_split
import prepprocessing
import MyTrain


# Загрузка и подготовка данных
df = prepprocessing.load_and_prepare("insurance.csv", transform='log')

# Разделение на признаки и таргет
X = df.drop("charges", axis=1)
y = df["charges"]

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Подбор параметров и обучение моделей
models = MyTrain.create_tuned_models(X_train, y_train)

# Оценка и обучение
MyTrain.train_and_evaluate(models, X_train, y_train)

#Построение и сохранение графиков остатков
MyTrain.plot_and_save_residuals(models, X_train, X_test, y_train, y_test, inverse_log=True)
