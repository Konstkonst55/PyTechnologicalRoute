def print_with_header(header: str, text: str):
    """
    Позволяет выводить в консоль текст с выделенным заголовком
    :param header: заголовок
    :param text: текст
    """
    header_accessory = "----------------"
    print(f"{header_accessory} {header} {header_accessory} \n {text}")
