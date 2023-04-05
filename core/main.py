
import os
import sys
import asyncio
import time

import pandas as pd
from art import tprint
from PyQt5 import QtWidgets
from pandas import DataFrame
from PyQt5.QtCore import QRegExp
from core.utils import Constants
from PyQt5.QtWidgets import QLineEdit
from ui.MainWindow import MainWindowUi
from PyQt5.QtGui import QRegExpValidator
from core.utils.FileReader import pick_file
from core.utils.RFRModel import get_rfr_model
from core.utils.MessageDisplayer import show_message
from core.processors.DataProcessor import DataProcessor
from core.processors.ModelProcessor import ModelProcessor
from core.utils.Validator import field_is_filled, values_is_float

# LabelEncoder dict
df_columns = ['name', 'gs_x', 'gs_y', 'gs_z', 'cg', 'mark', 'spf', 'tt']
osn_col_name = 'osn'
usl_col_name = 'usl'
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


# todo добавить ComboBox для текстовых данных из DataProcessor.le_dict


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model_processor = ModelProcessor(get_rfr_model(Constants.LOAD_MODEL_PATH))
        self.data_processor = DataProcessor()
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
        self.ui.b_learn_open.clicked.connect(self.learn_open_click)
        self.ui.b_info.clicked.connect(self.show_info_click)
        self.ui.b_ready.clicked.connect(self.predict_click)
        self.ui.b_predict_open.clicked.connect(self.predict_open_click)

    # функция нажатия на кнопку b_info
    @staticmethod
    def show_info_click():
        show_message(Constants.INFO_TEXT)

    # функция нажатия на кнопку b_ready
    def predict_click(self):
        try:
            print(len(self.model_processor.rfr_model.estimators_))

            if not field_is_filled(self.findChildren(QLineEdit)):
                show_message("Заполните все поля!")
                return

            field_value_list = [
                self.ui.le_gsy.text(),
                self.ui.le_gsx.text(),
                self.ui.le_gsz.text(),
                self.ui.le_cg.text()
            ]

            if not values_is_float(field_value_list):
                show_message("Проверьте поля!")
                return

            predict_df = self.model_processor.predict_data(
                self.get_predict_data_from_le(),
                osn_col_name,
                self.data_processor
            )

            self.ui.tb_output.setPlainText(
                str(self.data_processor.le_dict[osn_col_name]
                    .inverse_transform(
                        predict_df.astype(int).__getattr__(osn_col_name))
                    )
            )

        except AttributeError:
            show_message("Проверьте входной файл")
        except KeyError as ke:
            self.ui.tb_output.setPlainText(str(ke))
        except ValueError as ve:
            self.ui.tb_output.setPlainText(str(ve))

    # функция нажатия на кнопку b_learn_open
    def learn_open_click(self):
        try:
            file_name = pick_file()

            tprint("TR")

            if file_name != "":
                start_fit_time = time.time()
                self.ui.pb_fit_progress.setValue(24)

                x_trn, x_test, y_trn, y_test = asyncio.run(self.data_processor.process_data(file_name, osn_col_name))
                asyncio.run(self.model_processor.fit_model(x_trn, y_trn, x_test, y_test))

                end_fit_time = time.time()
                self.ui.l_info_tr.setText(f"{end_fit_time - start_fit_time:2f}s")
                self.ui.pb_fit_progress.setValue(100)

                show_message("Модель успешно обучена и сохранена!")
                self.ui.pb_fit_progress.setValue(0)
            else:
                self.model_processor.rfr_model = get_rfr_model(Constants.LOAD_MODEL_PATH)

        except KeyError as ke:
            self.ui.tb_output.setPlainText(str(ke))
            self.ui.pb_fit_progress.setValue(0)
        
    def predict_open_click(self):
        pass

    def get_predict_data_from_le(self) -> DataFrame:
        """
        Возвращает данные, введенные пользователем
        :return: возвращает данные с основными колонками, собранные из QLineEdit
        """
        return DataFrame(
            [[self.ui.le_name.text(),
              float(self.ui.le_gsx.text()),
              float(self.ui.le_gsy.text()),
              float(self.ui.le_gsz.text()),
              float(self.ui.le_cg.text()),
              self.ui.le_mark.text(),
              self.ui.le_spf.text(),
              self.ui.le_tt.toPlainText()]],
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
