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
