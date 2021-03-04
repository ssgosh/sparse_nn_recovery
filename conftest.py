import pytest
import logging as log


def pytest_assertion_pass(item, lineno, orig, expl):
    log.info("asserting that {}:{}, {}, {}".format(item, lineno, orig, expl))
