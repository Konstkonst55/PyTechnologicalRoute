from pathlib import Path

import pandas as pd
from PyQt5.QtWidgets import QFileDialog
from pandas import DataFrame

from core.utils import Constants

file_types_dict = {
    ".csv": pd.read_csv,
    ".json": pd.read_json
}


def pick_file() -> str:
    """
    Открывает проводник для выбора файла пользователем
    :return: полный путь до выбранного файла
    """
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    return file_dialog.getOpenFileName(
        caption=Constants.FILE_EXPL_HEADER,
        directory=Constants.FILE_EXPL_START_PATH,
        filter=Constants.FILE_EXPL_TYPES
    )[0]  # только путь до файла без типов


def read_data_file(file_name: str) -> DataFrame:
    """
    Позволяет читать файлы данных различного типа
    (*.csv *.json *.xlsx *.xls *.xml *.html)
    :param file_name: путь к файлу
    :return: DataFrame с данными
    """
    file_type = Path(file_name).suffix

    if file_type == ".csv":
        return pd.DataFrame(
            data=file_types_dict[file_type](file_name, encoding='utf-8', sep=';', engine='python')
        )
    else:
        return pd.DataFrame(
            data=file_types_dict[file_type](file_name)
        )
