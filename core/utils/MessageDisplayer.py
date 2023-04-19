from PyQt5.QtWidgets import QMessageBox
from sklearn.metrics import r2_score, mean_absolute_error


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
    print_with_header("Веса", model.feature_importances_)

    # точность
    print_with_header("Качество", model.score(x, y).round(2))

    # коэффициент детерминации
    print_with_header("R^2", r2_score(y, model.predict(x)).round(2))

    # средняя абсолютная ошибка
    print_with_header("MAE", mean_absolute_error(y, model.predict(x)).round(2))
