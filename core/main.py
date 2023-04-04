# system
import asyncio
import os
import sys
import time
# files
import pandas as pd
from pandas import DataFrame
# ui
from art import tprint
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRegExp
from ui.MainWindow import MainWindowUi
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QLineEdit
# sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
# utils
from core.utils import Constants
from core.utils.Validator import is_float
from core.utils.FileReader import read_data_file
from core.utils.PrintHelper import print_with_header
from core.utils.RFRModel import get_rfr_model, save_model

# LabelEncoder dict
le_dict = {}
df_columns = ['name', 'gs_x', 'gs_y', 'gs_z', 'cg', 'mark', 'spf', 'tt']
save_model_path = os.path.join(os.path.dirname(__file__), Constants.SAVE_MODEL_PATH)

# README
# На вход файлы подаются в виде массива со следующей последовательностью данных:
# uid(необязательно) name(str) gs_x(float) gs_y(float) gs_z(float) cg(float) osn(str) mark(str) spf(str) tt(str)
# tt - формируется из всех технических требований (для каждого uid) в одну строку (разделитель | - вертикальная черта)
# Массив данных для обучения и прогнозирования можно загрузить в виде .csv файла, сохраненного с разделителями ';'
# и кодировкой utf-8. В случае, если данные в полях отсутствуют, то пустые поля необходимо заменить на -1.

# Тестовые данные для прогнозирования (выход osn 002-053)
predict_test_data = DataFrame(
    [["Балка",
      261,
      0,
      0,
      90,
      "Д19ч",
      "Профиль",
      "Шероховатость поверхности указана цветом в соответствии с инструкцией 30.0011.0155.998 . "
      "| Предельные отклонения размеров, допуски формы и расположения поверхностей - по ОСТ 1 00022-80 . "
      "| Контроль визуальный - после анодирования . "
      "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
      "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
      "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
      "| Покрытие: Ан.Окс.нхр Эмаль ЭП-140, оранжевая. 597 ОСТ 1 90055-85 . "
      "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 . "
      "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 . "
      "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 . "
      "| Маркировать Чк и клеймить Кк шрифтом ПО-5 ГОСТ 2930-62 ."
      ]]
)


# todo добавить комбобоксы
# todo дописать док


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.rfp_model = get_rfr_model(Constants.LOAD_MODEL_PATH)
        self.ui = MainWindowUi()
        self.ui.setup_ui(self)
        self.init_ui()
        self.show()

    def init_ui(self):
        self.setFixedSize(580, 440)
        # Установка ограничений для текстового ввода
        reg_ex = QRegExp("[0-9]+.?[0-9]{,2}")
        self.ui.le_gsy.setValidator(QRegExpValidator(reg_ex, self.ui.le_gsy))
        self.ui.le_gsx.setValidator(QRegExpValidator(reg_ex, self.ui.le_gsx))
        self.ui.le_cg.setValidator(QRegExpValidator(reg_ex, self.ui.le_cg))
        self.ui.le_gsz.setValidator(QRegExpValidator(reg_ex, self.ui.le_gsz))
        # Установка методов для кнопок
        self.ui.b_learn_open.clicked.connect(self.pick_learn_file)
        self.ui.b_info.clicked.connect(show_info)
        self.ui.b_ready.clicked.connect(self.predict)

    def predict(self):
        try:
            print(len(self.rfp_model.estimators_))

            if not self.field_is_filled():
                show_message("Заполните все поля!")
                return

            if not self.field_is_valid():
                show_message("Проверьте поля!")
                return

            predict_df = predict_osn(self.rfp_model, self.get_predict_data_from_le())
            self.ui.tb_output.setPlainText(
                str(le_dict['osn'].inverse_transform(predict_df.astype(int).__getattr__('osn')))
            )
        except AttributeError:
            show_message("Сначала необходимо обучить модель")
        except KeyError as ke:
            self.ui.tb_output.setPlainText(f"le_dict не содержит данных для {str(ke)}")
        except ValueError as ve:
            self.ui.tb_output.setPlainText(f"le_dict не содержит данных для {str(ve)}")

    def pick_learn_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_name = file_dialog.getOpenFileName(
            self,
            caption=Constants.FILE_EXPL_HEADER,
            directory=Constants.FILE_EXPL_START_PATH,
            filter=Constants.FILE_EXPL_TYPES)

        tprint("TR")

        if file_name[0] != "":
            x_trn, x_test, y_trn, y_test = asyncio.run(process_data(file_name[0]))
            self.rfp_model = asyncio.run(fit_model(x_trn, y_trn, x_test, y_test))

            show_message("Модель успешно обучена и сохранена!")
            self.ui.pb_fit_progress.setValue(0)
        else:
            self.rfp_model = get_rfr_model(Constants.LOAD_MODEL_PATH)

    def field_is_filled(self) -> bool:
        line_edits = self.findChildren(QLineEdit)
        for le in line_edits:
            if le.text() == "":
                return False
        return True

    def field_is_valid(self) -> bool:
        return is_float(self.ui.le_gsy.text()) \
            and is_float(self.ui.le_gsx.text()) \
            and is_float(self.ui.le_gsz.text()) \
            and is_float(self.ui.le_cg.text())

    def get_predict_data_from_le(self) -> DataFrame:
        return DataFrame(
            [[self.ui.le_name.text(),
              float(self.ui.le_gsx.text()),
              float(self.ui.le_gsy.text()),
              float(self.ui.le_gsz.text()),
              float(self.ui.le_cg.text()),
              self.ui.le_mark.text(),
              self.ui.le_spf.text(),
              self.ui.le_tt.toPlainText()]]
        )


def show_message(text: str):
    msg = QMessageBox()
    msg.setText(text)
    msg.exec()


def show_info():
    show_message(Constants.INFO_TEXT)


def predict_osn(model, data: DataFrame):
    data.columns = df_columns

    try:
        # Прогнозирование цехов по входным данным
        predict = model.predict(transform_data(data))
        data.insert(5, 'osn', predict)
    except ValueError as ve:
        application.ui.tb_output.setPlainText(str(ve))

    return data


async def process_data(file_name: str):
    """
    Предназначена для преобразования входных данных по наименованию файла
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
        dataset = fit_transform_data(dataset)

        # Разбиение данных на прогнозируемые и обучающие
        trg_data = dataset[['osn']].values.ravel()
        trn_data = dataset.drop('osn', axis=1)

        # Разбиение данных на две выборки - тестовая и обучающая
        return train_test_split(trn_data, trg_data, test_size=0.1)
    except KeyError:
        show_message("Проверьте входной файл!")


async def fit_model(x_trn, y_trn, x_test, y_test):
    """
    Предназначена для обучения модели случайного леса
    :param x_trn: входная выборка тренировочных данных
    :param y_trn: входная выборка тренировочных данных
    :param x_test: входная выборка тестовых данных
    :param y_test: входная выборка тестовых данных
    :return: RandomForestRegressor
    """
    model = get_rfr_model(Constants.LOAD_MODEL_PATH)

    application.ui.pb_fit_progress.setValue(24)
    start_fit_time = time.time()

    model.fit(x_trn, y_trn)

    end_fit_time = time.time()
    elapsed_fit_time = end_fit_time - start_fit_time
    application.ui.l_info_tr.setText(f"{elapsed_fit_time:.2f}c")
    application.ui.pb_fit_progress.setValue(100)

    save_model(model, Constants.SAVE_MODEL_PATH)

    print_model_info(model, x_test, y_test)

    return model


def fit_transform_data(df: DataFrame):
    """
    Обучение и преобразование текстовых данных в числовой эквивалент по категориальным признакам
    :param df: входные данные
    :return: DataFrame
    """
    for col in df.columns:
        first = df.loc[df.index[0], col]
        if isinstance(first, str) or first == -1:
            le_dict[col] = LabelEncoder()
            df[col] = le_dict[col].fit_transform(df.astype(str).__getattr__(col))
    return df


def transform_data(df: DataFrame):
    """
    Преобразование текстовых данных в числовой эквивалент по обученным категориальным признакам
    :param df: входные данные
    :return: DataFrame
    """
    try:
        for col in df.columns:
            first = df.loc[df.index[0], col]
            if isinstance(first, str) or first == -1:
                df[col] = le_dict[col].transform(df.astype(str).__getattr__(col))
        return df
    except ValueError as ve:
        application.ui.tb_output.setPlainText(str(ve))
    except KeyError as ke:
        application.ui.tb_output.setPlainText(str(ke))


def print_model_info(model, x, y):
    """
    Позволяет выводить в консоль все данные о качестве модели
    (веса каждого параметра, общее качество, коэффициент детерминации, средняя абсолютная ошибка)
    :param model: модель
    :param x: тестовые данные x
    :param y: тестовые данные y
    """
    # веса
    print_with_header("Веса", model.feature_importances_)

    # точность
    print_with_header("Качество", model.score(x, y).round(2))

    # коэффициент детерминации
    print_with_header("R^2", r2_score(y, model.predict(x)).round(2))

    # средняя абсолютная ошибка
    print_with_header("MAE", mean_absolute_error(y, model.predict(x)).round(2))


if __name__ == "__main__":
    # Настройка вывода данных в консоль
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 500)

    # Работа с Qt приложением
    app = QtWidgets.QApplication(sys.argv)
    application = MainWindow()
    sys.exit(app.exec())
