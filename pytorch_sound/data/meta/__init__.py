"""
Meta Information Building
"""
import enum
import abc
from pandas import DataFrame
from typing import List


MetaType = enum.Enum('MetaType', 'audio_filename midi_filename speaker duration text')


class MetaFrame:

    @property
    def process_columns(self):
        fields = [MetaType.audio_filename, MetaType.midi_filename, MetaType.speaker, MetaType.text]
        return [col for col in self.columns if col in fields]

    @property
    @abc.abstractmethod
    def columns(self) -> List[MetaType]:
        raise NotImplementedError('You must define columns !')

    @property
    @abc.abstractmethod
    def meta(self) -> DataFrame:
        raise NotImplementedError('You must define make DataFrame!')

    @abc.abstractmethod
    def make_meta(self):
        raise NotImplementedError('You must define make DataFrame and save it !')
