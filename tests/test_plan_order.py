# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Every decision the job can make has its place in the plan's order, so a plan prints on every kind of machine."""

import inspect
import re

import pytest

from meepsquared import job as job_module


def test_every_decision_the_job_makes_is_in_the_order():
    source = inspect.getsource(job_module)
    named = set(re.findall(r'(?:Decision|_decide)\(\s*"([a-z_]+)"', source))
    assert named, "no decision names found in meepsquared.job"
    missing = sorted(named - set(job_module.ORDER))
    assert missing == [], f"decisions with no place in the plan: {missing}"


def test_an_unknown_decision_is_refused_with_its_name():
    with pytest.raises(RuntimeError, match="no place for the decisions \\['surprise'\\]"):
        job_module._ordered([job_module.Decision("surprise", "surprise", 1, "1")])
