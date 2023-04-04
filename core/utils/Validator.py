def is_float(obj):
    try:
        float(obj)
        return True
    except ValueError:
        return False
