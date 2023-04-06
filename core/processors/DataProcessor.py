from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from core.utils import Constants
from core.utils.FileReader import read_data_file
from core.utils.FileSaver import get_le, save
from core.utils.MessageDisplayer import print_with_header


class DataProcessor:
    """
    Класс предназначен для преобразования входных данных для дальнейшего обучения модели на них

    Атрибуты:
    --------
    le_dict = {}
        содержит несколько обученных LabelEncoder на различного типа данных

    Методы:
    -------

    Преобразование данных
    ---------------------
    process_data(file_name: str, predict_col_name: str)

    Трансформирование данных
    ------------------------
    Трансформирование данных производится по столбцам, которые содержат текстовые данные или -1

    fit_transform_data(df: DataFrame) -> DataFrame
        трансформирует текстовые данные, заменяя их во входном массиве
        (при этом происходит обучение кодировщика)
        чтобы трансформировать данные обратно, необходимо использовать следующий метод:

    transform_data(df: DataFrame) -> DataFrame
        трансформирует текстовые данные, на которых уже обучены кодировщики.
        Если LabelEncoder никогда не встречал входную строку, то выбрасывается исключение
    """
    def __init__(self):
        self.le_dict = get_le(Constants.ENCODER_PATH, Constants.ENCODER_FILE_NAME)

    async def process_data(self, file_name: str, predict_col_name: str):
        """
        Предназначена для преобразования входных данных по наименованию файла
        :param predict_col_name: наименование колонки прогнозируемых данных
        :param file_name: входной путь до файла .csv или .json
        :return: x_trn, x_test, y_trn, y_test
        """
        try:
            print_with_header("Входной файл", file_name)
            dataset = read_data_file(file_name)
            print_with_header("Набор данных", str(dataset))
            print_with_header("Корреляция", str(dataset.corr(numeric_only=True)))

            # Удаление идентификаторов, т.к. он не участвуют в обучении
            if 'uid' in dataset.columns:
                dataset = dataset.drop('uid', axis=1)

            # Обучение кодировщика на всех входных данных и преобразование данных
            dataset = self.fit_transform_data(dataset)

            # Разбиение данных на прогнозируемые и обучающие
            trg_data = dataset[[predict_col_name]].values.ravel()
            trn_data = dataset.drop(predict_col_name, axis=1)

            # Разбиение данных на две выборки - тестовая и обучающая
            return train_test_split(trn_data, trg_data, test_size=0.1)

        except KeyError as ke:
            raise KeyError(f"Проверьте входной файл {str(ke)}")

    def fit_transform_data(self, df: DataFrame) -> DataFrame:
        """
        Обучение и преобразование текстовых данных в числовой эквивалент по категориальным признакам
        :param df: входные данные
        :return: преобразованные данные
        """
        for col in df.columns:
            first = df.loc[df.index[0], col]
            if isinstance(first, str) or first == -1:
                self.le_dict[col] = LabelEncoder()
                df[col] = self.le_dict[col].fit_transform(df.astype(str).__getattr__(col))
        save(self.le_dict, Constants.ENCODER_PATH, Constants.ENCODER_FILE_NAME)
        return df

    def transform_data(self, df: DataFrame) -> DataFrame:
        """
        Преобразование текстовых данных в числовой эквивалент по обученным категориальным признакам
        :param df: входные данные
        :return: преобразованные данные
        """
        try:
            for col in df.columns:
                first = df.loc[df.index[0], col]
                if isinstance(first, str) or first == -1:
                    df[col] = self.le_dict[col].transform(df.astype(str).__getattr__(col))
            return df

        except ValueError as ve:
            raise ValueError(f"Неизвестные входные данные {str(ve)}")
        except KeyError as ke:
            raise KeyError(f"Неизвестные входные данные {str(ke)}")
