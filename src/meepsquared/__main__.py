# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""python -m meepsquared: the meepsquared command (see meepsquared.cli)."""

import sys

from meepsquared.cli import main

sys.exit(main())
