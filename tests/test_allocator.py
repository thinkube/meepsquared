# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""When meepsquared sets PyTorch's allocator to expandable segments, and when it leaves it alone."""

from meepsquared._allocator import SETTING, decide


def test_set_when_nothing_else_decided():
    environ = {}
    assert decide(environ, cuda_started=False, unsupported="") == "expandable segments"
    assert environ["PYTORCH_ALLOC_CONF"] == SETTING


def test_a_setting_of_the_user_stands():
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        environ = {name: "max_split_size_mb:512"}
        assert decide(environ, cuda_started=False, unsupported="") == f"as set in {name}"
        assert environ == {name: "max_split_size_mb:512"}


def test_once_cuda_started_the_running_allocator_is_changed():
    environ, applied = {}, []
    assert decide(environ, cuda_started=True, unsupported="", set_running=applied.append) == "expandable segments"
    assert applied == [SETTING] and environ["PYTORCH_ALLOC_CONF"] == SETTING


def test_before_cuda_starts_only_the_environment_is_set():
    applied = []
    decide({}, cuda_started=False, unsupported="", set_running=applied.append)
    assert applied == []


def test_not_set_where_unsupported():
    environ = {}
    assert decide(environ, cuda_started=False, unsupported="WSL") == "off (not supported on WSL)"
    assert environ == {}


def test_the_decision_is_made_on_import():
    import meepsquared
    from meepsquared import _allocator

    assert _allocator.DECISION != "not decided"
    assert meepsquared.__version__
