from typing import Final

FILE_EXPL_HEADER: Final[str] = "Выберите файл данных"
FILE_EXPL_START_PATH: Final[str] = "data"
FILE_EXPL_TYPES: Final[str] = "Text data files (*.csv *.json)"  # todo add *.xlsx *.xls *.xml *.html
MODEL_PATH: Final[str] = "data/models/"
MODEL_FILE_NAME: Final[str] = "model_details.joblib"
ENCODER_PATH: Final[str] = "data/encoders/"
ENCODER_FILE_NAME: Final[str] = "label_encoders_dict.joblib"
REGEX_FLOAT_TYPE: Final[str] = "[0-9]+.?[0-9]{,2}"
INFO_TEXT: Final[str] = ("tt - формируется из всех технических требований (для каждого uid) в одну строку "
                         "(разделитель | - вертикальная черта)\n\nМассив данных для обучения и прогнозирования "
                         "можно загрузить в виде .csv файла, сохраненного с разделителями ';' и кодировкой utf-8. "
                         "В случае, если данные в полях отсутствуют, то пустые поля необходимо заменить на -1."
                         "\n\nЕсли ComboBox'ы (name, mark, spf, tt) не содержат элементов, то необходимо "
                         "обучить модель")
