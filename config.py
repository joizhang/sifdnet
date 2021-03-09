import os

from dotenv import dotenv_values

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config(dict):

    def __init__(self):
        super().__init__()
        assert os.path.exists(os.path.join(BASE_DIR, '.env'))
        # print('Importing environment from .env file')
        env_values = dotenv_values(os.path.join(BASE_DIR, '.env'))
        # print(env_values)
        for key in env_values:
            if key.isupper():
                self[key] = env_values[key]

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, dict.__repr__(self))


if __name__ == '__main__':
    config = Config()
    print(config)
