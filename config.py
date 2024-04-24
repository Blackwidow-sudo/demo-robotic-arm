from dotenv import dotenv_values

values = dotenv_values(".env")

def get(key):
    return values.get(key, None)
