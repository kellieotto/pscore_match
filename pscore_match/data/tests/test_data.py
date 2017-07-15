from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import pscore_match.data as data
from numpy.testing import assert_equal

def test_ggi():
    """ Test that "Gerber-Green-Imai" data can be loaded. """
    ggi = data.gerber_green_imai()
    assert_equal(ggi.shape, (10829, 26))


def test_dw():
    """ Test that "Dehejia-Wahba" data can be loaded. """
    dw = data.dehejia_wahba()
    assert_equal(dw.shape, (445, 10))
    