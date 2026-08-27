from abc import ABCMeta


class Data(metaclass=ABCMeta):

    def __init__(self, name: str, verbose: int = 0):
        self.name = name
        self.verbose = verbose
