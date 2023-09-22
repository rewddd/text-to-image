import os
from pathlib import Path

import yaml
from pytest import Metafunc

from .helpers import BaseModelTest, BaseModelTestCase

test_cases_directory = Path(os.path.dirname(os.path.realpath(__file__))) / "artifacts"


def load_model_tests(model_name: str):
    test_cases_yaml_path = (test_cases_directory / model_name).with_suffix(".yaml")

    with open(test_cases_yaml_path) as fp:
        test_cases = yaml.safe_load(fp)

    for test_case in test_cases:
        yield BaseModelTestCase(**test_case)


def sanitize_model_name(model_name: str) -> str:
    lowered_model_name = model_name.lower()
    return lowered_model_name.replace("_", "").replace("test", "")


def get_model_name_from_class(cls: type) -> str:
    if hasattr(cls, "model_name"):
        return cls.model_name
    else:
        return sanitize_model_name(cls.__name__)


def pytest_generate_tests(metafunc: Metafunc):
    if metafunc.cls and issubclass(metafunc.cls, BaseModelTest):
        model_name = get_model_name_from_class(metafunc.cls)

        if "model_test_case" in metafunc.fixturenames:
            metafunc.parametrize("model_test_case", load_model_tests(model_name))
