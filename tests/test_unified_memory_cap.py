# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""On a GPU that shares the system's memory, PyTorch's allocator is capped below what the system has available."""

import pytest

from meepsquared import _unified_memory

GIB = 2**30


def test_the_cap_is_the_available_memory_less_four_gib():
    assert _unified_memory.allocator_cap(106 * GIB, 121 * GIB) == 102 * GIB
    assert _unified_memory.allocator_cap(50 * GIB, 64 * GIB) == 46 * GIB


def test_a_machine_with_less_than_the_headroom_left_is_refused_with_the_figures():
    with pytest.raises(RuntimeError, match="3.0 GiB available of 64.0 GiB"):
        _unified_memory.allocator_cap(3 * GIB, 64 * GIB)


def test_available_memory_reads_the_kernel():
    assert _unified_memory.available_memory() > 0
