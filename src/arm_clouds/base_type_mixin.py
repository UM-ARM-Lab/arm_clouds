from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic, Optional, Iterable, Iterator
from collections import UserList, UserDict

import torch


class BaseTypeMixin(ABC):
    """The base type mix-in that all CDCPD-types inherit from.

    NOTE: All type containers should inherit from either UserList or UserDict *and* BaseTypeMixin.
    This ensures that all containers for types behave as lists or dicts and have the methods
    required for interacting with them defined in this class.
    """
    is_torch: bool

    @abstractmethod
    def to_torch(self, dtype: torch.dtype = torch.double, device: torch.device = "cpu"):
        pass

    @abstractmethod
    def to_numpy(self):
        pass

    @abstractmethod
    def copy(self):
        pass


class BaseTypeDictMixin(UserDict, ABC):
    """The base type mix-in that all CDCPD-types inherit from.

    NOTE: All type dicts should inherit from UserDict *and* BaseTypeMixin.
    This ensures that all containers for types behave as dicts and have the methods
    required for interacting with them defined in this class.
    """

    @abstractmethod
    def to_torch(self):
        pass
