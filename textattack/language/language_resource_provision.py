"""
Language Resource Provision
===========================

Language resource provision objects provide a generic interface for resource requests that are language dependent.

This module implements as base class that is also a singleton. Users can define their own derived classes if their
language does not offer a resource provision class yet.

The user instanciates the language singelton according to the language for which resources need to be requested in down
stream logic.

Example:

    set_language('Japanese')
    EmbeddingAugmenter().augment(...)

    TODO: this requires the `EmbeddingAugmenter` and all relveant downstream components to make use of the language
        singleton to request the language specific resources.
"""

from abc import ABCMeta
from .languages import *
import os


class SingletonABCMeta(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(SingletonABCMeta, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class LanguageResourceProvision(metaclass=SingletonABCMeta):
    pass


def set_language(language: str):
    # normalize language name: Old English -> OldEnglish, french -> French, ...
    language_norm = ''.join(part.title() for part in language.split())
    try:
        eval(language_norm + 'ResourceProvison()')
    except NameError:
        raise NameError(f'No language resource provision class `{language_norm}` defined for "{language}". Add it to {os.path.join(os.path.basename(__file__), "languages")}')
