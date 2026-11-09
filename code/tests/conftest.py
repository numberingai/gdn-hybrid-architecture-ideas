import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)
