"""Shared test fixtures."""

import numpy as np
import torch
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def torch_rng():
    return torch.Generator().manual_seed(42)


@pytest.fixture
def grid_shape():
    return (32, 32)


@pytest.fixture
def out_shape():
    return (28, 28)


@pytest.fixture
def n_coils():
    return 4


@pytest.fixture
def n_samples():
    return 256
