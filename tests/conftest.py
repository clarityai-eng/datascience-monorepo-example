import os
import sys

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "fixtures/data",
)
SRC_DIR = os.path.realpath("src")


def pytest_configure(config):
    sys.path.append("src")
