from pandas import DataFrame

from core.processors.DataProcessor import DataProcessor
from core.utils import Constants
from core.utils.FileSaver import get_rfr_model, save
from core.utils.MessageDisplayer import print_model_info


# todo upd doc

class ModelProcessor:
    """
    Класс предназначен для работы с моделью, ее обучением и прогнозированием данных

    Атрибуты:
    --------
    rfr_model_osn: RandomForestRegressor
        модель с основными цехами

    rfr_model_usl: RandomForestRegressor
        модель с цехами услуг

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
    def __init__(self, model_osn, model_usl):
        self.rfr_model_osn = model_osn
        self.rfr_model_usl = model_usl

    async def fit_osn_model(self, x_trn, y_trn, x_test, y_test):
        """
        Предназначена для обучения модели случайного леса основных цехов
        :param x_trn: входная выборка тренировочных данных
        :param y_trn: входная выборка тренировочных данных
        :param x_test: входная выборка тестовых данных
        :param y_test: входная выборка тестовых данных
        :return: модель RandomForestRegressor
        """
        self.rfr_model_osn = get_rfr_model(Constants.MODEL_OSN_PATH, Constants.MODEL_OSN_FILE_NAME)
        self.rfr_model_osn.fit(x_trn, y_trn)
        save(self.rfr_model_osn, Constants.MODEL_OSN_PATH, Constants.MODEL_OSN_FILE_NAME)
        print_model_info(self.rfr_model_osn, x_test, y_test)

    async def fit_usl_model(self, x_trn, y_trn, x_test, y_test):
        """
        Предназначена для обучения модели случайного леса цехов-услуг
        :param x_trn: входная выборка тренировочных данных
        :param y_trn: входная выборка тренировочных данных
        :param x_test: входная выборка тестовых данных
        :param y_test: входная выборка тестовых данных
        :return: модель RandomForestRegressor
        """
        self.rfr_model_usl = get_rfr_model(Constants.MODEL_USL_PATH, Constants.MODEL_USL_FILE_NAME)
        self.rfr_model_usl.fit(x_trn, y_trn)
        save(self.rfr_model_usl, Constants.MODEL_USL_PATH, Constants.MODEL_USL_FILE_NAME)
        print_model_info(self.rfr_model_usl, x_test, y_test)

    def predict_data(self, data: DataFrame, predict_col_name: str, data_processor: DataProcessor) -> DataFrame:
        """
        Выполняет прогнозирование данных по входному набору и формирует выходной набор
        с исходными данными + предсказанные данные для каждой строки
        :param data_processor: входной преобразователь данных с обученными LabelEncoders
        :param predict_col_name: наименование колонки прогнозируемых данных
        :param data: данные для прогнозирования
        :return: исходные данные + предсказанные данные для каждой строки
        """
        if predict_col_name == "osn":
            predict = self.rfr_model_osn.predict(data_processor.transform_osn_data(data))
        elif predict_col_name == "usl":
            predict = self.rfr_model_usl.predict(data_processor.transform_usl_data(data))
        else:
            raise ValueError(f"Ошибка в наименовании прогнозируемой колонки {predict_col_name}")

        data.insert(5, predict_col_name, predict)
        return data
