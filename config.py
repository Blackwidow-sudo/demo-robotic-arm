from dotenv import dotenv_values

values = dotenv_values(".env")

def get(key, type=str):
    value = values.get(key, None)

    return value if value is None else type(value)
