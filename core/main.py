import asyncio
import sys
import time

import pandas as pd
from PyQt5 import QtWidgets
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator
from PyQt5.QtWidgets import QLineEdit, QListView
from art import tprint
from pandas import DataFrame

from core.processors.DataProcessor import DataProcessor
from core.processors.ModelProcessor import ModelProcessor
from core.utils import Constants
from core.utils.FileReader import pick_file, read_data_file
from core.utils.FileSaver import get_rfc_model
from core.utils.MessageDisplayer import show_message
from core.utils.Validator import field_is_filled, values_is_float
from ui.MainWindow import MainWindowUi

df_columns = ['name', 'gab', 'cg', 'mark', 'spf', 'tt', 'agr']
osn_col_name = 'osn'
usl_col_name = 'usl'

# README
# На вход файлы подаются в виде массива со следующей последовательностью данных:
# uid(необязательно) name(str) gab(float) cg(float) osn(str) mark(str) spf(str) tt(str) agr(float)
# tt - формируется из всех технических требований (для каждого uid) в одну строку (разделитель | - вертикальная черта)
# Массив данных для обучения и прогнозирования можно загрузить в виде .csv файла, сохраненного с разделителями ';'
# и кодировкой utf-8. В случае, если данные в полях отсутствуют, то пустые поля необходимо заменить на -1.


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_processor = ModelProcessor(
            get_rfc_model(Constants.MODEL_OSN_PATH, Constants.MODEL_OSN_FILE_NAME),
            get_rfc_model(Constants.MODEL_USL_PATH, Constants.MODEL_USL_FILE_NAME)
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
        reg_ex = QRegExp(Constants.REGEX_FLOAT_TYPE)
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
        show_message(Constants.INFO_TEXT)

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
