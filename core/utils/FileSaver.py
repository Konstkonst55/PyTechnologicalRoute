import os
from operator import concat
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor


def get_rfr_model(path: str, name: str) -> RandomForestRegressor:
    """
    Получение обученной модели Случайного леса, если такая есть по указанному пути.
    Иначе создается новая модель
    :param name: имя получаемой модели
    :param path: путь к модели
    :return: RandomForestRegressor
    """
    full_path = f"{path}{name}"

    if Path(full_path).exists():
        return joblib.load(full_path)
    else:
        print("get_rfr_model вернул RandomForestRegressor")
        return RandomForestRegressor()


def get_le(path: str, name: str) -> dict:
    """
    Получение обученного dict с LabelEncoders, если такие есть по указанному пути.
    Иначе возвращается пустой dict
    :param name: имя получаемого dict
    :param path: путь к LabelEncoder
    :return: dict
    """
    full_path = f"{path}{name}"

    if Path(full_path).exists():
        return joblib.load(full_path)
    else:
        print("get_le вернул {}")
        return {}


def save(value, path: str, name: str):
    """
    Сохраняет файл модели по указанному пути
    :param name: имя сохраняемого файла
    :param value: any
    :param path: путь для сохранения
    """
    if Path(path).exists():
        joblib.dump(value, concat(path, name))
    else:
        raise FileExistsError(f"Путь {Path(path).absolute()}, указанный при сохранении файла {name} отсутствует")
