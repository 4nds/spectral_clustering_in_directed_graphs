from __future__ import annotations

import datetime
from abc import abstractmethod
from typing import Any, Collection, Protocol, TypeVar, Union, overload

import numpy as np

# pylint: disable-next=invalid-name
ScalarType_co = TypeVar('ScalarType_co', bound=np.generic, covariant=True)
MaskedNDArray = np.ma.MaskedArray[Any, np.dtype[ScalarType_co]]

NumpyBoolConvertible = bool
NumpyBoolLike = np.bool_ | NumpyBoolConvertible

NumpyIntegerConvertible = int | datetime.timedelta
NumpyIntegerLike = np.integer | NumpyIntegerConvertible
NumpyFloatingConvertible = float
NumpyFloatingLike = np.floating | NumpyFloatingConvertible
NumpyComplexFloatingConvertible = complex
NumpyComplexFloatingLike = np.complexfloating | NumpyComplexFloatingConvertible
NumpyInexactConvertible = (
    NumpyFloatingConvertible | NumpyComplexFloatingConvertible
)
NumpyInexactLike = np.inexact | NumpyInexactConvertible
NumpyNumberConvertible = Union[
    NumpyIntegerConvertible, NumpyInexactConvertible
]
NumpyNumberLike = np.integer | NumpyNumberConvertible

NumpyFlexibleConvertible = bytes | str
NumpyFlexibleLike = np.flexible | NumpyFlexibleConvertible

NumpyDatetimeConvertible = datetime.datetime
NumpyDatetimeLike = np.datetime64 | NumpyDatetimeConvertible

NumpyGenericConvertible = (
    NumpyBoolConvertible
    | NumpyNumberConvertible
    | NumpyFlexibleConvertible
    | NumpyDatetimeConvertible
)
NumpyGenericLike = np.generic | NumpyGenericConvertible

NumpyGenericLikeT = TypeVar(
    'NumpyGenericLikeT',
    np.generic,
    bool,
    int,
    datetime.timedelta,
    float,
    complex,
    bytes,
    str,
    datetime.datetime,
)

T_co = TypeVar('T_co', covariant=True)

class SimpleSequence(Collection[T_co], Protocol[T_co]):
    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> T_co: ...
    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> SimpleSequence[T_co]: ...
