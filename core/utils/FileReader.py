from pathlib import Path

import pandas as pd

file_types_dict = {
    ".csv": pd.read_csv,
    ".json": pd.read_json
}


def read_data_file(file_name: str):
    """
    Позволяет читать файлы данных различного типа
    (*.csv *.json *.xlsx *.xls *.xml *.html)
    :param file_name: путь к файлу
    :return: DataFrame
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
