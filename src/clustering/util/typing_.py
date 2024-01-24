import datetime
from typing import Any, TypeVar, Union

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
