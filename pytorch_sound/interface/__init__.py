import abc


class Interface:
    """
    Defines the interface between 'wav' and 'model'
    """

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError()
