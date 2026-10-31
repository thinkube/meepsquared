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


def test_job_cores_is_the_affinity_within_the_cgroup_quota(monkeypatch, tmp_path):
    monkeypatch.setattr(_unified_memory.os, "sched_getaffinity", lambda pid: set(range(16)))
    real_open = open

    def opening(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/cpu.max":
            return real_open(tmp_path / "cpu.max", *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", opening)
    (tmp_path / "cpu.max").write_text("400000 100000\n")
    assert _unified_memory.job_cores() == 4
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert _unified_memory.job_cores() == 16
    (tmp_path / "cpu.max").write_text("50000 100000\n")
    assert _unified_memory.job_cores() == 1


def test_job_cores_without_a_cgroup_file_is_the_affinity(monkeypatch):
    monkeypatch.setattr(_unified_memory.os, "sched_getaffinity", lambda pid: set(range(6)))
    real_open = open

    def opening(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/cpu.max":
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", opening)
    assert _unified_memory.job_cores() == 6
