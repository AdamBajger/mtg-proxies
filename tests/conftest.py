import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def cache_dir():
    return Path(".test_cache")
