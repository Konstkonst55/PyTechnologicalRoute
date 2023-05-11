import asyncio
import sys
import time
from operator import concat
from pathlib import Path
from typing import Final

import joblib
import pandas as pd
import pkg_resources
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QLineEdit, QListView, QFileDialog, QMessageBox
from art import tprint
from pandas import DataFrame
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df_columns = ['name', 'gab', 'cg', 'mark', 'spf', 'tt', 'agr']
osn_col_name = 'osn'
usl_col_name = 'usl'
file_types_dict = {
    ".csv": pd.read_csv,
    ".json": pd.read_json
}

FILE_EXPL_HEADER: Final[str] = "Выберите файл данных"
FILE_EXPL_START_PATH: Final[str] = "data"
FILE_EXPL_TYPES: Final[str] = "Text data files (*.csv *.json)"  # todo add *.xlsx *.xls *.xml *.html

MODEL_OSN_PATH = "data/models/osn/"
ENCODER_OSN_PATH = "data/encoders/osn/"
MODEL_USL_PATH = "data/models/usl/"
ENCODER_USL_PATH = "data/encoders/usl/"
MODEL_OSN_FILE_NAME: Final[str] = "model_osn_details.joblib"
ENCODER_OSN_FILE_NAME: Final[str] = "label_encoders_osn_dict.joblib"
MODEL_USL_FILE_NAME: Final[str] = "model_usl_details.joblib"
ENCODER_USL_FILE_NAME: Final[str] = "label_encoders_usl_dict.joblib"

REGEX_FLOAT_TYPE: Final[str] = "[0-9]+.?[0-9]{,2}"
INFO_TEXT: Final[str] = ("tt - формируется из всех технических требований (для каждого uid) в одну строку "
                         "(разделитель ' | ' - вертикальная черта и пробелы)\n\nМассив данных для обучения и "
                         "прогнозирования можно загрузить в виде .csv файла, сохраненного с разделителями ';' и "
                         "кодировкой utf-8. В случае, если данные в полях отсутствуют, то пустые поля необходимо "
                         "заменить на -1. \n\nЕсли ComboBox'ы (name, mark, spf, tt) не содержат элементов, "
                         "то необходимо обучить модель")


def is_float(obj) -> bool:
    """
    Проверка на то, является ли значение float или нет
    :param obj: входное значение
    :return: True, если obj - float
    False, если obj - не float
    """
    try:
        float(obj)
        return True
    except ValueError:
        return False


def field_is_filled(line_edits: list) -> bool:
    """
    Проверка на заполненность всех текстовых полей типа QLineEdit
    (или других объектов, содержащих атрибут 'text')
    :return: True, если все текстовые поля заполнены
    False, если хотя бы одно поле не заполнено
    """
    attr = "text"

    for le in line_edits:
        if not hasattr(le, attr):
            raise AttributeError(f"Поле {type(le)} не содержит атрибута {attr}")

        if le.text() == "":
            return False

    return True


def values_is_float(values: list) -> bool:
    """
    Проверка значений на корректность ввода
    :return: True, если все значения - числа (float)
    False, если хотя бы одно значение не число (float)
    """
    for val in values:
        if not is_float(val):
            return False
    return True


def print_with_header(header: str, text: str):
    """
    Позволяет выводить в консоль текст с выделенным заголовком
    :param header: заголовок
    :param text: текст
    """
    header_accessory = "-^35"
    print(f"{header:{header_accessory}} \n {text}")


def show_message(text: str):
    """
    Показывает диалоговое окно с указанным текстом
    :param text: текст диалогового окна
    """
    msg = QMessageBox()
    msg.setText(text)
    msg.exec()


def print_model_info(model, x, y):
    """
    Позволяет выводить в консоль все данные о качестве модели
    (веса каждого параметра, общее качество, коэффициент детерминации, средняя абсолютная ошибка)
    :param model: модель
    :param x: тестовые данные x
    :param y: тестовые данные y
    """
    # веса
    print_with_header(
        "Веса",
        str([round(elem, 2) for elem in model.feature_importances_])
    )

    # точность
    print_with_header("Качество", model.score(x, y).round(2))

    # коэффициент детерминации
    print_with_header("R^2", r2_score(y, model.predict(x)).round(2))

    # средняя абсолютная ошибка
    print_with_header("MAE", mean_absolute_error(y, model.predict(x)).round(2))


def get_rfc_model(path: str, name: str) -> RandomForestClassifier:
    """
    Получение обученной модели Случайного леса, если такая есть по указанному пути.
    Иначе создается новая модель
    :param name: имя получаемой модели
    :param path: путь к модели
    :return: RandomForestClassifier
    """
    full_path = f"{path}{name}"

    if Path(full_path).exists():
        return joblib.load(full_path)
    else:
        print("get_rfr_model вернул RandomForestClassifier")
        return RandomForestClassifier(n_estimators=150)


def get_le(path: str, name: str) -> dict:
    """
    Получение обученного dict с LabelEncoders, если такие есть по указанному пути.
    Иначе возвращается пустой dict
    :param name: имя получаемого dict
    :param path: путь к LabelEncoder
    :return: dict
    """
    full_path = f"{path}{name}"

    if Path(full_path).exists():
        return joblib.load(full_path)
    else:
        print("get_le вернул {}")
        return {}


def save(value, path: str, name: str):
    """
    Сохраняет файл модели по указанному пути
    :param name: имя сохраняемого файла
    :param value: any
    :param path: путь для сохранения
    """
    if Path(path).exists():
        joblib.dump(value, concat(path, name))
    else:
        raise FileExistsError(f"Путь {Path(path).absolute()}, указанный при сохранении файла {name} отсутствует")


def pick_file() -> str:
    """
    Открывает проводник для выбора файла пользователем
    :return: полный путь до выбранного файла
    """
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.ExistingFile)
    return file_dialog.getOpenFileName(
        caption=FILE_EXPL_HEADER,
        directory=FILE_EXPL_START_PATH,
        filter=FILE_EXPL_TYPES
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


# README
# На вход файлы подаются в виде массива со следующей последовательностью данных:
# uid(необязательно) name(str) gab(float) cg(float) osn(str) mark(str) spf(str) tt(str) agr(float)
# tt - формируется из всех технических требований (для каждого uid) в одну строку (разделитель | - вертикальная черта)
# Массив данных для обучения и прогнозирования можно загрузить в виде .csv файла, сохраненного с разделителями ';'
# и кодировкой utf-8. В случае, если данные в полях отсутствуют, то пустые поля необходимо заменить на -1.


class MainWindowUi(object):
    def setup_ui(self, main_window):
        main_window.setObjectName("TechRoutePredictor")
        main_window.resize(732, 675)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(main_window.sizePolicy().hasHeightForWidth())
        main_window.setSizePolicy(sizePolicy)
        self.central = QtWidgets.QWidget(main_window)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.central.sizePolicy().hasHeightForWidth())
        self.central.setSizePolicy(sizePolicy)
        self.central.setObjectName("central")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.central)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.tabs = QtWidgets.QTabWidget(self.central)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.tabs.setFont(font)
        self.tabs.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabs.setStyleSheet("\n"
                                "border: 0px solid black;\n"
                                "")
        self.tabs.setMovable(True)
        self.tabs.setTabBarAutoHide(True)
        self.tabs.setObjectName("tabs")
        self.tab_osn_usl = QtWidgets.QWidget()
        self.tab_osn_usl.setObjectName("tab_osn_usl")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_osn_usl)
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName("verticalLayout")
        self.hl_top = QtWidgets.QHBoxLayout()
        self.hl_top.setObjectName("hl_top")
        self.vl_type = QtWidgets.QVBoxLayout()
        self.vl_type.setContentsMargins(-1, -1, 10, -1)
        self.vl_type.setObjectName("vl_type")
        self.rb_osn = QtWidgets.QRadioButton(self.tab_osn_usl)
        self.rb_osn.setChecked(True)
        self.rb_osn.setObjectName("rb_osn")
        self.vl_type.addWidget(self.rb_osn)
        self.rb_usl = QtWidgets.QRadioButton(self.tab_osn_usl)
        self.rb_usl.setObjectName("rb_usl")
        self.vl_type.addWidget(self.rb_usl)
        self.hl_top.addLayout(self.vl_type)
        self.b_learn_open = QtWidgets.QPushButton(self.tab_osn_usl)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_learn_open.sizePolicy().hasHeightForWidth())
        self.b_learn_open.setSizePolicy(sizePolicy)
        self.b_learn_open.setMinimumSize(QtCore.QSize(0, 40))
        self.b_learn_open.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_learn_open.setFont(font)
        self.b_learn_open.setStyleSheet("QPushButton{\n"
                                        "border-radius: 5px;\n"
                                        "border: 1px solid #2B2A29;\n"
                                        "padding: 10px 10px 10px 10px;\n"
                                        "color: black;\n"
                                        "}\n"
                                        "\n"
                                        "QPushButton:hover{\n"
                                        "background: solid #625B71;\n"
                                        "color: white;\n"
                                        "}")
        self.b_learn_open.setObjectName("b_learn_open")
        self.hl_top.addWidget(self.b_learn_open, 0, QtCore.Qt.AlignVCenter)
        self.b_predict_open = QtWidgets.QPushButton(self.tab_osn_usl)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_predict_open.sizePolicy().hasHeightForWidth())
        self.b_predict_open.setSizePolicy(sizePolicy)
        self.b_predict_open.setMinimumSize(QtCore.QSize(0, 40))
        self.b_predict_open.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_predict_open.setFont(font)
        self.b_predict_open.setStyleSheet("QPushButton{\n"
                                          "border-radius: 5px;\n"
                                          "border: 1px solid #2B2A29;\n"
                                          "padding: 10px 10px 10px 10px;\n"
                                          "color: black;\n"
                                          "}\n"
                                          "\n"
                                          "QPushButton:hover{\n"
                                          "background: solid #625B71;\n"
                                          "color: white;\n"
                                          "}")
        self.b_predict_open.setObjectName("b_predict_open")
        self.hl_top.addWidget(self.b_predict_open, 0, QtCore.Qt.AlignVCenter)
        self.b_info = QtWidgets.QToolButton(self.tab_osn_usl)
        self.b_info.setMinimumSize(QtCore.QSize(40, 40))
        self.b_info.setMaximumSize(QtCore.QSize(40, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_info.setFont(font)
        self.b_info.setToolTipDuration(-1)
        self.b_info.setStyleSheet("border-radius: 5px;\n"
                                  "border: 1px solid #2B2A29;\n"
                                  "color: black;\n"
                                  "")
        self.b_info.setObjectName("b_info")
        self.hl_top.addWidget(self.b_info, 0, QtCore.Qt.AlignVCenter)
        self.l_info_tr = QtWidgets.QLabel(self.tab_osn_usl)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.l_info_tr.sizePolicy().hasHeightForWidth())
        self.l_info_tr.setSizePolicy(sizePolicy)
        self.l_info_tr.setMinimumSize(QtCore.QSize(40, 40))
        self.l_info_tr.setMaximumSize(QtCore.QSize(16777215, 40))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_info_tr.setFont(font)
        self.l_info_tr.setAcceptDrops(False)
        self.l_info_tr.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.l_info_tr.setStyleSheet("border-radius: 5px;\n"
                                     "border: 1px solid #2B2A29;\n"
                                     "padding: 10px 10px 10px 10px;\n"
                                     "color: black;\n"
                                     "")
        self.l_info_tr.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignTrailing | QtCore.Qt.AlignVCenter)
        self.l_info_tr.setObjectName("l_info_tr")
        self.hl_top.addWidget(self.l_info_tr, 0, QtCore.Qt.AlignVCenter)
        self.hl_top.setStretch(1, 1)
        self.hl_top.setStretch(2, 1)
        self.verticalLayout.addLayout(self.hl_top)
        self.top_line = QtWidgets.QFrame(self.tab_osn_usl)
        self.top_line.setStyleSheet("background: #2B2A29;")
        self.top_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.top_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.top_line.setObjectName("top_line")
        self.verticalLayout.addWidget(self.top_line)
        self.hl_text_top = QtWidgets.QHBoxLayout()
        self.hl_text_top.setObjectName("hl_text_top")
        self.vl_name = QtWidgets.QVBoxLayout()
        self.vl_name.setObjectName("vl_name")
        self.l_name = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_name.setFont(font)
        self.l_name.setObjectName("l_name")
        self.vl_name.addWidget(self.l_name)
        self.cb_name = QtWidgets.QComboBox(self.tab_osn_usl)
        self.cb_name.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_name.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_name.setFont(font)
        self.cb_name.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                   "border-radius: 5px;\n"
                                   "background: #D1D2FD solid;\n"
                                   "color: #2B2A29;\n"
                                   "")
        self.cb_name.setEditable(True)
        self.cb_name.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_name.setIconSize(QtCore.QSize(20, 20))
        self.cb_name.setFrame(True)
        self.cb_name.setObjectName("cb_name")
        self.vl_name.addWidget(self.cb_name)
        self.hl_text_top.addLayout(self.vl_name)
        self.vl_x = QtWidgets.QVBoxLayout()
        self.vl_x.setObjectName("vl_x")
        self.l_gab = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_gab.setFont(font)
        self.l_gab.setObjectName("l_gab")
        self.vl_x.addWidget(self.l_gab)
        self.le_gab = QtWidgets.QLineEdit(self.tab_osn_usl)
        self.le_gab.setMinimumSize(QtCore.QSize(0, 30))
        self.le_gab.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.le_gab.setFont(font)
        self.le_gab.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.le_gab.setInputMethodHints(QtCore.Qt.ImhHiddenText)
        self.le_gab.setObjectName("le_gab")
        self.vl_x.addWidget(self.le_gab)
        self.hl_text_top.addLayout(self.vl_x)
        self.vl_cg = QtWidgets.QVBoxLayout()
        self.vl_cg.setObjectName("vl_cg")
        self.l_cg = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_cg.setFont(font)
        self.l_cg.setObjectName("l_cg")
        self.vl_cg.addWidget(self.l_cg)
        self.le_cg = QtWidgets.QLineEdit(self.tab_osn_usl)
        self.le_cg.setMinimumSize(QtCore.QSize(0, 30))
        self.le_cg.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.le_cg.setFont(font)
        self.le_cg.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                 "border-radius: 5px;\n"
                                 "background: #D1D2FD solid;\n"
                                 "color: #2B2A29;\n"
                                 "")
        self.le_cg.setInputMethodHints(QtCore.Qt.ImhNone)
        self.le_cg.setObjectName("le_cg")
        self.vl_cg.addWidget(self.le_cg)
        self.hl_text_top.addLayout(self.vl_cg)
        self.hl_text_top.setStretch(0, 3)
        self.hl_text_top.setStretch(1, 1)
        self.hl_text_top.setStretch(2, 1)
        self.verticalLayout.addLayout(self.hl_text_top)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.l_spf = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_spf.setFont(font)
        self.l_spf.setObjectName("l_spf")
        self.verticalLayout_3.addWidget(self.l_spf)
        self.cb_spf = QtWidgets.QComboBox(self.tab_osn_usl)
        self.cb_spf.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_spf.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_spf.setFont(font)
        self.cb_spf.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.cb_spf.setEditable(True)
        self.cb_spf.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_spf.setIconSize(QtCore.QSize(20, 20))
        self.cb_spf.setFrame(True)
        self.cb_spf.setObjectName("cb_spf")
        self.verticalLayout_3.addWidget(self.cb_spf)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.l_agr = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_agr.setFont(font)
        self.l_agr.setObjectName("l_agr")
        self.verticalLayout_4.addWidget(self.l_agr)
        self.cb_agr = QtWidgets.QComboBox(self.tab_osn_usl)
        self.cb_agr.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_agr.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_agr.setFont(font)
        self.cb_agr.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                  "border-radius: 5px;\n"
                                  "background: #D1D2FD solid;\n"
                                  "color: #2B2A29;\n"
                                  "")
        self.cb_agr.setEditable(True)
        self.cb_agr.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_agr.setIconSize(QtCore.QSize(20, 20))
        self.cb_agr.setFrame(True)
        self.cb_agr.setObjectName("cb_agr")
        self.verticalLayout_4.addWidget(self.cb_agr)
        self.horizontalLayout.addLayout(self.verticalLayout_4)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.l_mark = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_mark.setFont(font)
        self.l_mark.setObjectName("l_mark")
        self.verticalLayout.addWidget(self.l_mark)
        self.cb_mark = QtWidgets.QComboBox(self.tab_osn_usl)
        self.cb_mark.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_mark.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_mark.setFont(font)
        self.cb_mark.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                   "border-radius: 5px;\n"
                                   "background: #D1D2FD solid;\n"
                                   "color: #2B2A29;\n"
                                   "")
        self.cb_mark.setEditable(True)
        self.cb_mark.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_mark.setIconSize(QtCore.QSize(20, 20))
        self.cb_mark.setFrame(True)
        self.cb_mark.setObjectName("cb_mark")
        self.verticalLayout.addWidget(self.cb_mark)
        self.l_tt = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_tt.setFont(font)
        self.l_tt.setObjectName("l_tt")
        self.verticalLayout.addWidget(self.l_tt)
        self.cb_tt = QtWidgets.QComboBox(self.tab_osn_usl)
        self.cb_tt.setMinimumSize(QtCore.QSize(0, 30))
        self.cb_tt.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.cb_tt.setFont(font)
        self.cb_tt.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cb_tt.setStyleSheet("border: 1px solid #8F8FB0;\n"
                                 "border-radius: 5px;\n"
                                 "background: #D1D2FD solid;\n"
                                 "color: #2B2A29;\n"
                                 "")
        self.cb_tt.setEditable(True)
        self.cb_tt.setCurrentText("")
        self.cb_tt.setInsertPolicy(QtWidgets.QComboBox.InsertAtTop)
        self.cb_tt.setIconSize(QtCore.QSize(20, 20))
        self.cb_tt.setFrame(True)
        self.cb_tt.setObjectName("cb_tt")
        self.verticalLayout.addWidget(self.cb_tt)
        self.b_ready = QtWidgets.QPushButton(self.tab_osn_usl)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.b_ready.sizePolicy().hasHeightForWidth())
        self.b_ready.setSizePolicy(sizePolicy)
        self.b_ready.setMaximumSize(QtCore.QSize(16777215, 41))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_ready.setFont(font)
        self.b_ready.setStyleSheet("QPushButton{\n"
                                   "border-radius: 5px;\n"
                                   "border: 1px solid #2B2A29;\n"
                                   "color: black;\n"
                                   "padding: 10px 10px 10px 10px;\n"
                                   "}\n"
                                   "\n"
                                   "QPushButton:hover{\n"
                                   "background: solid #625B71;\n"
                                   "color: white;\n"
                                   "}")
        self.b_ready.setObjectName("b_ready")
        self.verticalLayout.addWidget(self.b_ready, 0, QtCore.Qt.AlignRight)
        self.hl_out_header = QtWidgets.QHBoxLayout()
        self.hl_out_header.setObjectName("hl_out_header")
        self.l_out = QtWidgets.QLabel(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.l_out.setFont(font)
        self.l_out.setObjectName("l_out")
        self.hl_out_header.addWidget(self.l_out)
        self.b_clear_in = QtWidgets.QPushButton(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_clear_in.setFont(font)
        self.b_clear_in.setStyleSheet("QPushButton{\n"
                                      "border-radius: 5px;\n"
                                      "border: 1px solid #2B2A29;\n"
                                      "color: black;\n"
                                      "padding: 3px 10px 3px 10px;\n"
                                      "}\n"
                                      "\n"
                                      "QPushButton:hover{\n"
                                      "background: solid #625B71;\n"
                                      "color: white;\n"
                                      "}")
        self.b_clear_in.setObjectName("b_clear_in")
        self.hl_out_header.addWidget(self.b_clear_in, 0, QtCore.Qt.AlignRight)
        self.b_clear_out = QtWidgets.QPushButton(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.b_clear_out.setFont(font)
        self.b_clear_out.setStyleSheet("QPushButton{\n"
                                       "border-radius: 5px;\n"
                                       "border: 1px solid #2B2A29;\n"
                                       "color: black;\n"
                                       "padding: 3px 10px 3px 10px;\n"
                                       "}\n"
                                       "\n"
                                       "QPushButton:hover{\n"
                                       "background: solid #625B71;\n"
                                       "color: white;\n"
                                       "}")
        self.b_clear_out.setObjectName("b_clear_out")
        self.hl_out_header.addWidget(self.b_clear_out, 0, QtCore.Qt.AlignRight)
        self.hl_out_header.setStretch(0, 5)
        self.verticalLayout.addLayout(self.hl_out_header)
        self.bottom_line = QtWidgets.QFrame(self.tab_osn_usl)
        self.bottom_line.setStyleSheet("background: #2B2A29;")
        self.bottom_line.setFrameShape(QtWidgets.QFrame.HLine)
        self.bottom_line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.bottom_line.setObjectName("bottom_line")
        self.verticalLayout.addWidget(self.bottom_line)
        self.tb_output = QtWidgets.QPlainTextEdit(self.tab_osn_usl)
        self.tb_output.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tb_output.sizePolicy().hasHeightForWidth())
        self.tb_output.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.tb_output.setFont(font)
        self.tb_output.setStyleSheet("border-radius: 5px;\n"
                                     "border: 1px solid  rgb(155, 241, 56);\n"
                                     "color: black;\n"
                                     "")
        self.tb_output.setReadOnly(True)
        self.tb_output.setPlainText("")
        self.tb_output.setObjectName("tb_output")
        self.verticalLayout.addWidget(self.tb_output)
        self.pb_fit_progress = QtWidgets.QProgressBar(self.tab_osn_usl)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semilight")
        font.setPointSize(9)
        self.pb_fit_progress.setFont(font)
        self.pb_fit_progress.setStyleSheet("border-radius: 5px;\n"
                                           "border: 1px solid  rgb(155, 241, 56);\n"
                                           "color: black;\n"
                                           "")
        self.pb_fit_progress.setProperty("value", 0)
        self.pb_fit_progress.setAlignment(QtCore.Qt.AlignCenter)
        self.pb_fit_progress.setObjectName("pb_fit_progress")
        self.verticalLayout.addWidget(self.pb_fit_progress)
        self.tabs.addTab(self.tab_osn_usl, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.tabs.addTab(self.tab_2, "")
        self.verticalLayout_2.addWidget(self.tabs)
        main_window.setCentralWidget(self.central)
        self.usl = QtWidgets.QAction(main_window)
        self.usl.setObjectName("usl")

        self.retranslate_ui(main_window)
        self.tabs.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(main_window)

    def retranslate_ui(self, main_window):
        _translate = QtCore.QCoreApplication.translate
        main_window.setWindowTitle(_translate("main_window", "MainWindow"))
        self.rb_osn.setText(_translate("main_window", "Осн."))
        self.rb_usl.setText(_translate("main_window", "Усл."))
        self.b_learn_open.setText(_translate("main_window", "Выбрать данные для обучения"))
        self.b_predict_open.setText(_translate("main_window", "Выбрать данные для прогнозирования"))
        self.b_info.setToolTip(_translate("main_window", "Информация"))
        self.b_info.setText(_translate("main_window", "..."))
        self.l_info_tr.setToolTip(_translate("main_window", "Время обучения модели"))
        self.l_info_tr.setText(_translate("main_window", "0"))
        self.l_name.setText(_translate("main_window", "Наименование"))
        self.cb_name.setToolTip(_translate("main_window", "Наименование"))
        self.l_gab.setText(_translate("main_window", "Габариты"))
        self.le_gab.setToolTip(_translate("main_window", "Габариты (0-4)"))
        self.l_cg.setText(_translate("main_window", "Констр. группа"))
        self.le_cg.setToolTip(_translate("main_window", "Конструктивная группа (две цифры)"))
        self.l_spf.setText(_translate("main_window", "Полуфабрикат"))
        self.cb_spf.setToolTip(_translate("main_window", "Полуфабрикат"))
        self.l_agr.setText(_translate("main_window", "Агрегат"))
        self.cb_agr.setToolTip(_translate("main_window", "Агрегат"))
        self.l_mark.setText(_translate("main_window", "Марка"))
        self.cb_mark.setToolTip(_translate("main_window", "Марка"))
        self.l_tt.setText(_translate("main_window", "Технические требования"))
        self.cb_tt.setToolTip(_translate("main_window", "Технические требования"))
        self.b_ready.setText(_translate("main_window", "Готово"))
        self.l_out.setText(_translate("main_window", "Вывод"))
        self.b_clear_in.setText(_translate("main_window", "Очистить ввод"))
        self.b_clear_out.setText(_translate("main_window", "Очистить вывод"))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_osn_usl), _translate("main_window", "Осн. / Усл."))
        self.tabs.setTabText(self.tabs.indexOf(self.tab_2), _translate("main_window", "Другое"))
        self.usl.setText(_translate("main_window", "usl"))


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
        self.le_dict_osn = get_le(ENCODER_OSN_PATH, ENCODER_OSN_FILE_NAME)
        self.le_dict_usl = get_le(ENCODER_USL_PATH, ENCODER_USL_FILE_NAME)

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
            print_with_header("Типы параметров", str(dataset.dtypes))

            # Удаление идентификаторов, т.к. он не участвуют в обучении
            if 'uid' in dataset.columns:
                dataset = dataset.drop('uid', axis=1)

            # Обучение кодировщика на всех входных данных и преобразование данных
            if predict_col_name == "osn":
                dataset = self.fit_transform_osn_data(dataset)
            elif predict_col_name == "usl":
                dataset = self.fit_transform_usl_data(dataset)
            else:
                raise ValueError(f"Ошибка в наименовании прогнозируемой колонки {predict_col_name}")

            # Разбиение данных на прогнозируемые и обучающие
            trg_data = dataset[[predict_col_name]].values.ravel()
            trn_data = dataset.drop(predict_col_name, axis=1)

            # Разбиение данных на две выборки - тестовая и обучающая
            return train_test_split(trn_data, trg_data, test_size=0.1)

        except KeyError as ke:
            raise KeyError(f"Проверьте входной файл {str(ke)}")

    def fit_transform_osn_data(self, df: DataFrame) -> DataFrame:
        """
        Обучение и преобразование текстовых данных в числовой эквивалент по категориальным признакам основных цехов
        :param df: входные данные
        :return: преобразованные данные
        """
        df = df.sort_values(by="osn")

        for col in df.columns:
            if df[col].dtype == object or df[col].eq(-1).any():
                self.le_dict_osn[col] = LabelEncoder()
                df[col] = self.le_dict_osn[col].fit_transform(df.astype(str).__getattr__(col))
        save(self.le_dict_osn, ENCODER_OSN_PATH, ENCODER_OSN_FILE_NAME)
        return df

    def transform_osn_data(self, df: DataFrame) -> DataFrame:
        """
        Преобразование текстовых данных в числовой эквивалент по обученным категориальным признакам основных цехов
        :param df: входные данные
        :return: преобразованные данные
        """
        try:
            for col in df.columns:
                if df[col].dtype == object or df[col].eq(-1).any():
                    df[col] = self.le_dict_osn[col].transform(df.astype(str).__getattr__(col))
            return df

        except ValueError as ve:
            raise ValueError(f"Неизвестные входные данные {str(ve)}")
        except KeyError as ke:
            raise KeyError(f"Неизвестные входные данные {str(ke)}")

    def fit_transform_usl_data(self, df: DataFrame) -> DataFrame:
        """
        Обучение и преобразование текстовых данных в числовой эквивалент по категориальным признакам цехов-услуг
        :param df: входные данные
        :return: преобразованные данные
        """
        df = df.sort_values(by="usl")

        for col in df.columns:
            if df[col].dtype == object or df[col].eq(-1).any():
                self.le_dict_usl[col] = LabelEncoder()
                df[col] = self.le_dict_usl[col].fit_transform(df.astype(str).__getattr__(col))
        save(self.le_dict_usl, ENCODER_USL_PATH, ENCODER_USL_FILE_NAME)
        return df

    def transform_usl_data(self, df: DataFrame) -> DataFrame:
        """
        Преобразование текстовых данных в числовой эквивалент по обученным категориальным признакам цехов-услуг
        :param df: входные данные
        :return: преобразованные данные
        """
        try:
            for col in df.columns:
                if df[col].dtype == object or df[col].eq(-1).any():
                    df[col] = self.le_dict_usl[col].transform(df.astype(str).__getattr__(col))
            return df

        except ValueError as ve:
            raise ValueError(f"Неизвестные входные данные {str(ve)}")
        except KeyError as ke:
            raise KeyError(f"Неизвестные входные данные {str(ke)}")

    def inverse_transform_osn_data(self, df: DataFrame) -> DataFrame:
        """

        :param df:
        :return:
        """
        pass

    def inverse_transform_usl_data(self, df: DataFrame) -> DataFrame:
        """

        :param df:
        :return:
        """
        pass


class ModelProcessor:
    """
    Класс предназначен для работы с моделью, ее обучением и прогнозированием данных

    Атрибуты:
    --------
    rfr_model_osn: RandomForestClassifier
        модель с основными цехами

    rfr_model_usl: RandomForestClassifier
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
        :return: модель RandomForestClassifier
        """
        self.rfr_model_osn = get_rfc_model(MODEL_OSN_PATH, MODEL_OSN_FILE_NAME)
        self.rfr_model_osn.fit(x_trn, y_trn)
        save(self.rfr_model_osn, MODEL_OSN_PATH, MODEL_OSN_FILE_NAME)
        print_model_info(self.rfr_model_osn, x_test, y_test)

    async def fit_usl_model(self, x_trn, y_trn, x_test, y_test):
        """
        Предназначена для обучения модели случайного леса цехов-услуг
        :param x_trn: входная выборка тренировочных данных
        :param y_trn: входная выборка тренировочных данных
        :param x_test: входная выборка тестовых данных
        :param y_test: входная выборка тестовых данных
        :return: модель RandomForestClassifier
        """
        self.rfr_model_usl = get_rfc_model(MODEL_USL_PATH, MODEL_USL_FILE_NAME)
        self.rfr_model_usl.fit(x_trn, y_trn)
        save(self.rfr_model_usl, MODEL_USL_PATH, MODEL_USL_FILE_NAME)
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_processor = ModelProcessor(
            get_rfc_model(MODEL_OSN_PATH, MODEL_OSN_FILE_NAME),
            get_rfc_model(MODEL_USL_PATH, MODEL_USL_FILE_NAME)
        )
        self.data_processor = DataProcessor()
        self.ui = MainWindowUi()
        self.ui.setup_ui(self)
        self.cb_list = {self.ui.cb_name, self.ui.cb_mark, self.ui.cb_spf, self.ui.cb_tt, self.ui.cb_agr}
        self.cb_dict = {
            self.ui.cb_name: "name",
            self.ui.cb_mark: "mark",
            self.ui.cb_spf: "spf",
            self.ui.cb_tt: "tt",
            self.ui.cb_agr: "agr"
        }
        self.type_is_osn = True
        self.init_ui()
        self.load_cb_data()
        self.show()

    def init_ui(self):
        # Установка ListView как представление для ComboBox tt
        listView = QListView()
        listView.setWordWrap(True)
        self.ui.cb_tt.setView(listView)

        # Установка ограничений для текстового ввода
        reg_ex = QRegExp(REGEX_FLOAT_TYPE)
        self.ui.le_gab.setValidator(QRegExpValidator(reg_ex, self.ui.le_gab))
        self.ui.le_cg.setValidator(QRegExpValidator(reg_ex, self.ui.le_cg))

        # Установка методов для кнопок
        self.ui.b_learn_open.clicked.connect(self.learn_open_click)
        self.ui.b_info.clicked.connect(self.show_info_click)
        self.ui.b_ready.clicked.connect(self.predict_click)
        self.ui.b_predict_open.clicked.connect(self.predict_open_click)
        self.ui.b_clear_in.clicked.connect(self.clear_in_click)
        self.ui.b_clear_out.clicked.connect(self.clear_out_click)
        self.ui.rb_osn.clicked.connect(self.on_type_switch_click)
        self.ui.rb_usl.clicked.connect(self.on_type_switch_click)

    def on_type_switch_click(self):
        self.type_is_osn = self.ui.rb_osn.isChecked()
        self.load_cb_data()

    def clear_in_click(self):
        for cb in self.cb_list:
            cb.setCurrentIndex(-1)

        for le in self.findChildren(QLineEdit):
            le.setText("")

    def clear_out_click(self):
        self.ui.tb_output.setPlainText("")

    def load_cb_data(self):
        if self.type_is_osn:
            if len(self.data_processor.le_dict_osn.items()) < 1:
                self.ui.tb_output.setPlainText("Необходимо обучить модель osn")
                return

            for cb in self.cb_list:
                cb.clear()
                cb.addItems(self.data_processor.le_dict_osn[self.cb_dict[cb]].classes_)
                cb.setCurrentIndex(-1)
        else:
            if len(self.data_processor.le_dict_usl.items()) < 1:
                self.ui.tb_output.setPlainText("Необходимо обучить модель usl")
                return

            for cb in self.cb_list:
                cb.clear()
                cb.addItems(self.data_processor.le_dict_usl[self.cb_dict[cb]].classes_)
                cb.setCurrentIndex(-1)

        self.clear_in_click()

    # функция нажатия на кнопку b_info
    @staticmethod
    def show_info_click():
        show_message(INFO_TEXT)

    # функция нажатия на кнопку b_ready
    def predict_click(self):
        try:
            if not field_is_filled(self.findChildren(QLineEdit)):
                show_message("Заполните все поля!")
                return

            field_value_list = [
                self.ui.le_gab.text(),
                self.ui.le_cg.text()
            ]

            if not values_is_float(field_value_list):
                show_message("Проверьте поля!")
                return

            if self.type_is_osn:
                if not hasattr(self.model_processor.rfr_model_osn, "estimators_"):
                    raise AttributeError("Сначала необходимо обучить модель osn")

                predict_df = self.model_processor.predict_data(
                    self.get_predict_data_from_le(),
                    osn_col_name,
                    self.data_processor
                )

                self.ui.tb_output.setPlainText(
                    str(
                        self.data_processor.le_dict_osn[osn_col_name]
                        .inverse_transform(
                            predict_df.astype(int).__getattr__(osn_col_name)
                        )[0]
                    )
                )
            else:
                if not hasattr(self.model_processor.rfr_model_usl, "estimators_"):
                    raise AttributeError("Сначала необходимо обучить модель usl")

                predict_df = self.model_processor.predict_data(
                    self.get_predict_data_from_le(),
                    usl_col_name,
                    self.data_processor
                )

                self.ui.tb_output.setPlainText(
                    str(
                        self.data_processor.le_dict_usl[usl_col_name]
                        .inverse_transform(
                            predict_df.astype(int).__getattr__(usl_col_name)
                        )
                    )
                )

        except AttributeError as ae:
            self.ui.tb_output.setPlainText(str(ae))
        except KeyError as ke:
            self.ui.tb_output.setPlainText(str(ke))
        except ValueError as ve:
            self.ui.tb_output.setPlainText(str(ve))

    # функция нажатия на кнопку b_learn_open
    def learn_open_click(self):
        try:
            file_name = pick_file()

            if file_name == "":
                return

            start_fit_time = time.time()
            self.ui.pb_fit_progress.setValue(24)

            tprint("TR")

            if self.type_is_osn:
                x_trn, x_test, y_trn, y_test = asyncio.run(self.data_processor.process_data(file_name, osn_col_name))
                asyncio.run(self.model_processor.fit_osn_model(x_trn, y_trn, x_test, y_test))
            else:
                x_trn, x_test, y_trn, y_test = asyncio.run(self.data_processor.process_data(file_name, usl_col_name))
                asyncio.run(self.model_processor.fit_usl_model(x_trn, y_trn, x_test, y_test))

            self.load_cb_data()

            end_fit_time = time.time()
            self.ui.l_info_tr.setText(f"{(end_fit_time - start_fit_time):.2f}s")
            self.ui.pb_fit_progress.setValue(100)
            show_message("Модель успешно обучена и сохранена!")
            self.ui.pb_fit_progress.setValue(0)

        except KeyError as ke:
            self.ui.tb_output.setPlainText(str(ke))
            self.ui.pb_fit_progress.setValue(0)
        except FileExistsError as fee:
            self.ui.tb_output.setPlainText(str(fee))
            self.ui.pb_fit_progress.setValue(0)
        except ValueError as ve:
            self.ui.tb_output.setPlainText(str(ve))
            self.ui.pb_fit_progress.setValue(0)

    def predict_open_click(self):
        try:
            file_name = pick_file()

            if file_name == "":
                return

            df = read_data_file(file_name)

            if self.type_is_osn:
                if not hasattr(self.model_processor.rfr_model_osn, "estimators_"):
                    raise AttributeError("Сначала необходимо обучить модель osn")

                predict_df = self.model_processor.predict_data(
                    df,
                    osn_col_name,
                    self.data_processor
                )
                # todo use inverse_transform_osn_data()
                predict_df[osn_col_name] = self.data_processor.le_dict_osn[osn_col_name]\
                    .inverse_transform(predict_df.astype(int).__getattr__(osn_col_name))

                self.ui.tb_output.setPlainText(str(predict_df))
            else:
                if not hasattr(self.model_processor.rfr_model_usl, "estimators_"):
                    raise AttributeError("Сначала необходимо обучить модель usl")

                predict_df = self.model_processor.predict_data(
                    df,
                    usl_col_name,
                    self.data_processor
                )
                # todo use inverse_transform_usl_data(df)
                predict_df[usl_col_name] = self.data_processor.le_dict_usl[usl_col_name]\
                    .inverse_transform(predict_df.astype(int).__getattr__(usl_col_name))

                self.ui.tb_output.setPlainText(str(predict_df))

        except KeyError as ke:
            self.ui.tb_output.setPlainText(str(ke))
        except FileExistsError as fee:
            self.ui.tb_output.setPlainText(str(fee))
        except ValueError as ve:
            self.ui.tb_output.setPlainText(str(ve))
        except AttributeError as ae:
            self.ui.tb_output.setPlainText(str(ae))

    def get_predict_data_from_le(self) -> DataFrame:
        """
        Возвращает данные, введенные пользователем
        :return: возвращает данные с основными колонками, собранные из QLineEdit
        """
        return DataFrame(
            [[self.ui.cb_name.currentText(),
              float(self.ui.le_gab.text()),
              float(self.ui.le_cg.text()),
              self.ui.cb_mark.currentText(),
              self.ui.cb_spf.currentText(),
              self.ui.cb_tt.currentText(),
              self.ui.cb_agr.currentText()]],
            columns=df_columns
        )


if __name__ == "__main__":
    # Настройка вывода данных в консоль
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 500)

    MODEL_OSN_PATH = pkg_resources.resource_filename(__name__, MODEL_OSN_PATH)
    ENCODER_OSN_PATH = pkg_resources.resource_filename(__name__, ENCODER_OSN_PATH)
    MODEL_USL_PATH = pkg_resources.resource_filename(__name__, MODEL_USL_PATH)
    ENCODER_USL_PATH = pkg_resources.resource_filename(__name__, ENCODER_USL_PATH)

    # Работа с Qt приложением
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    sys.exit(app.exec())

"""

 _           _   
| |_ ___ ___| |_ 
| '_|   |_ -|  _|
|_,_|_|_|___|_|  

"""
