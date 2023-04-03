# system
import sys
# files
import pandas as pd
from pandas import DataFrame
# ui
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from ui.MainWindow import Ui_MainWindow
# sklearn
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# utils
from core.utils import Constants
from core.utils.FileReader import read_data_file
from core.utils.PrintHelper import print_with_header
from core.utils.RFRModel import get_rfr_model, save_model
# other
from art import tprint


# LabelEncoder dict
le_dict = {}

# названия колонок данных, которые нужно трансформировать
data_cols_name_list = ['name', 'osn', 'mark', 'spf', 'tt']
predict_cols_name_list = ['name', 'mark', 'spf', 'tt']

sample_weights = [.3, .7, .7, .7, .5, 1, .3]  # todo настроить веса

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

# todo сделать интерфейс ввода
# todo добавить автоматическое определение текстовых столбцов


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_ui()
        self.show()

    def init_ui(self):
        self.ui.b_file_open.clicked.connect(self.pick_file)

    def pick_file(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_name = file_dialog.getOpenFileName(
            self,
            caption=Constants.FILE_EXPL_HEADER,
            directory=Constants.FILE_EXPL_START_PATH,
            filter=Constants.FILE_EXPL_TYPES)

        tprint("TR")

        process_data(file_name[0])  # file_name[0] - полное имя файла file_name[1] - типы файлов


def process_data(file_name: str):
    try:
        if file_name != "":
            print_with_header("Входной файл", file_name)
            dataset = read_data_file(file_name)
            print_with_header("Набор данных", str(dataset))
            print_with_header("Корреляция", str(dataset.corr(numeric_only=True)))

            # Удаление идентификаторов, т.к. они не участвуют в обучении
            if 'uid' in dataset.columns.names:
                dataset = dataset.drop('uid', axis=1)

            # Обучение кодировщика на всех входных данных и преобразование данных
            dataset = fit_transform_data(dataset, data_cols_name_list)

            # Разбиение данных на прогнозируемые и обучающие
            trg_data = dataset[['osn']].values.ravel()
            trn_data = dataset.drop('osn', axis=1)

            predict_test_data.columns = trn_data.columns

            # Разбиение данных на две выборки - тестовая и обучающая
            x_trn, x_test, y_trn, y_test = train_test_split(trn_data, trg_data, test_size=0.1)

            print_with_header("x_trn", str(DataFrame(x_trn)))
            print_with_header("x_test", str(DataFrame(x_test)))
            print_with_header("y_trn", str(DataFrame(y_trn)))
            print_with_header("y_test", str(DataFrame(y_test)))

            fit_model(
                x_trn=x_trn,
                y_trn=y_trn,
                x_test=x_test,
                y_test=y_test
            )
    except KeyError as ke:
        print(ke)


def fit_model(x_trn, y_trn, x_test, y_test):
    # Обучение модели случайного леса
    model = get_rfr_model(Constants.MODEL_PATH)

    model.fit(x_trn, y_trn)

    save_model(model, Constants.MODEL_PATH)

    print_model_info(model, x_test, y_test)

    # Прогнозирование цехов по входным данным
    predict = model.predict(transform_data(predict_test_data, predict_cols_name_list))
    predict_test_data.insert(5, 'osn', int(predict))

    # Вывод спрогнозированного результата в приложении
    application.ui.l_head_text.setText(
        f"Предсказание тест\n"
        f"{le_dict['osn'].inverse_transform(predict_test_data['osn'])}"
    )


def fit_transform_data(df: DataFrame, col_names: list[str]):
    """
    Обучение и преобразование текстовых данных в числовой эквивалент по категориальным признакам
    :param df: входные данные
    :param col_names: список названий колонок, подлежащих обучению и преобразованию
    :return: DataFrame
    """
    for col in col_names:
        le_dict[col] = LabelEncoder()
        df[col] = le_dict[col].fit_transform(df.astype(str).__getattr__(col))
    return df


def transform_data(df: DataFrame, col_names: list[str]):
    """
        Преобразование текстовых данных в числовой эквивалент по обученным категориальным признакам
        :param df: входные данные
        :param col_names: список названий колонок, подлежащих преобразованию по обученным данным
        :return: DataFrame
        """
    for col in col_names:
        df[col] = le_dict[col].transform(df.astype(str).__getattr__(col))
    return df


def print_model_info(model, x, y):
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
