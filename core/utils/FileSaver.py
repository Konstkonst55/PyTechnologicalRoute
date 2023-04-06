import os
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor


def get_rfr_model(path: str):
    """
    Получение обученной модели Случайного леса, если такая есть по указанному пути.
    Иначе создается новая модель
    :param path: путь к модели
    :return: RandomForestRegressor
    """
    if Path(path).exists():
        return joblib.load(path)
    else:
        return RandomForestRegressor()


def save(value, path: str):
    """
    Сохраняет файл модели по указанному пути
    :param value: Any
    :param path: путь для сохранения
    """
    if Path(path).exists():
        joblib.dump(value, path)
    else:
        raise FileExistsError("Путь, указанный при сохранении файла отсутствует")
