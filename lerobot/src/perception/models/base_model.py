from abc import ABC, abstractmethod
import torch

class BaseModel(ABC):
    model_registry = {}

    @classmethod
    def register_model(cls, model_name):
        def wapper(subclass):
            cls.model_registry[model_name] = subclass
            return subclass

        return wapper

    @classmethod
    def create_model(cls, model_name, *args, **kwargs):
        if model_name not in cls.model_registry:
            raise ValueError(
                f"Model with key '{model_name}' is not registered."
            )
        task_class = cls.model_registry[model_name]
        return task_class(*args, **kwargs)
