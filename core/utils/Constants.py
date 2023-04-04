from typing import Final

FILE_EXPL_HEADER: Final[str] = "Выберите файл данных"
FILE_EXPL_START_PATH: Final[str] = "data"
FILE_EXPL_TYPES: Final[str] = "Text data files (*.csv *.json)"  # todo add *.xlsx *.xls *.xml *.html
LOAD_MODEL_PATH: Final[str] = "data\\model\\model_details.joblib"
SAVE_MODEL_PATH: Final[str] = LOAD_MODEL_PATH
INFO_TEXT: Final[str] = "tt - формируется из всех технических требований (для каждого uid) в одну строку " \
                        "(разделитель | - вертикальная черта)\nМассив данных для обучения и прогнозирования " \
                        "можно загрузить в виде .csv файла, сохраненного с разделителями ';' и кодировкой utf-8. " \
                        "В случае, если данные в полях отсутствуют, то пустые поля необходимо заменить на -1."
