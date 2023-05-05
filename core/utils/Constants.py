from typing import Final

FILE_EXPL_HEADER: Final[str] = "Выберите файл данных"
FILE_EXPL_START_PATH: Final[str] = "data"
FILE_EXPL_TYPES: Final[str] = "Text data files (*.csv *.json)"  # todo add *.xlsx *.xls *.xml *.html

MODEL_OSN_PATH: Final[str] = "data/models/osn/"
MODEL_OSN_FILE_NAME: Final[str] = "model_osn_details.joblib"
ENCODER_OSN_PATH: Final[str] = "data/encoders/osn/"
ENCODER_OSN_FILE_NAME: Final[str] = "label_encoders_osn_dict.joblib"
MODEL_USL_PATH: Final[str] = "data/models/usl/"
MODEL_USL_FILE_NAME: Final[str] = "model_usl_details.joblib"
ENCODER_USL_PATH: Final[str] = "data/encoders/usl/"
ENCODER_USL_FILE_NAME: Final[str] = "label_encoders_usl_dict.joblib"

REGEX_FLOAT_TYPE: Final[str] = "[0-9]+.?[0-9]{,2}"
INFO_TEXT: Final[str] = ("tt - формируется из всех технических требований (для каждого uid) в одну строку "
                         "(разделитель ' | ' - вертикальная черта и пробелы)\n\nМассив данных для обучения и "
                         "прогнозирования можно загрузить в виде .csv файла, сохраненного с разделителями ';' и "
                         "кодировкой utf-8. В случае, если данные в полях отсутствуют, то пустые поля необходимо "
                         "заменить на -1. \n\nЕсли ComboBox'ы (name, mark, spf, tt) не содержат элементов, "
                         "то необходимо обучить модель")
