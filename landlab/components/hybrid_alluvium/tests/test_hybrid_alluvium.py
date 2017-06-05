#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 12:59:44 2017

@author: barnhark
"""

from __future__ import print_function
import os

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

try:
    from nose.tools import assert_is
except ImportError:
    from landlab.testing.tools import assert_is