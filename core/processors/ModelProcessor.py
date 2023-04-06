from pandas import DataFrame

from core.processors.DataProcessor import DataProcessor
from core.utils import Constants
from core.utils.FileSaver import get_rfr_model, save
from core.utils.MessageDisplayer import print_model_info


class ModelProcessor:
    """
    Класс предназначен для работы с моделью, ее обучением и прогнозированием данных

    Атрибуты:
    --------
    rfr_model: RandomForestRegressor
        модель, которой производится прогнозирование

    Методы:
    -------

    Обучение модели
    ---------------
    fit_model(x_trn, y_trn, x_test, y_test)
        обучает модель на предоставленных данных

    Прогнозирование данных
    ----------------------
    predict_osn(data: DataFrame, predict_col_name: str, data_processor: DataProcessor)
        выполняет прогнозирование данных
    """
    def __init__(self, model):
        """
        :param model: модель, которой производится прогнозирование
        """
        self.rfr_model = model

    async def fit_model(self, x_trn, y_trn, x_test, y_test):
        """
        Предназначена для обучения модели случайного леса
        :param x_trn: входная выборка тренировочных данных
        :param y_trn: входная выборка тренировочных данных
        :param x_test: входная выборка тестовых данных
        :param y_test: входная выборка тестовых данных
        :return: модель RandomForestRegressor
        """
        self.rfr_model = get_rfr_model(Constants.MODEL_PATH, Constants.MODEL_FILE_NAME)

        self.rfr_model.fit(x_trn, y_trn)

        save(self.rfr_model, Constants.MODEL_PATH, Constants.MODEL_FILE_NAME)

        print_model_info(self.rfr_model, x_test, y_test)

    def predict_data(self, data: DataFrame, predict_col_name: str, data_processor: DataProcessor) -> DataFrame:
        """
        Выполняет прогнозирование данных по входному набору и формирует выходной набор
        с исходными данными + предсказанные данные для каждой строки
        :param data_processor: входной преобразователь данных с обученными LabelEncoders
        :param predict_col_name: наименование колонки прогнозируемых данных
        :param data: данные для прогнозирования
        :return: исходные данные + предсказанные данные для каждой строки
        """
        try:
            predict = self.rfr_model.predict(data_processor.transform_data(data))
            data.insert(5, predict_col_name, predict)
            return data

        except ValueError as ve:
            raise ValueError(f"Неизвестные входные данные {str(ve)}")
