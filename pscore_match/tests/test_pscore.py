from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises
import numpy as np
import pandas as pd

from ..pscore import PropensityScore
from ..data import gerber_green_imai

def test_imai_data():
    imai = gerber_green_imai()
    # Create interaction terms
    imai['PERSONS1_VOTE961'] = (imai.PERSONS==1)*imai.VOTE961
    imai['PERSONS1_NEW'] = (imai.PERSONS==1)*imai.NEW
    treatment = np.array(imai.PHNC1)
    cov_list = ['PERSONS', 'VOTE961', 'NEW', 'MAJORPTY', 'AGE', 'WARD', 'AGE2',
				'PERSONS1_VOTE961', 'PERSONS1_NEW']
    covariates = imai[cov_list]
    pscore = PropensityScore(treatment, covariates).compute()
    np.testing.assert_almost_equal(np.mean(pscore[treatment==1]), 0.03, decimal=1)
    np.testing.assert_almost_equal(np.mean(pscore[treatment==0]), 0.02, decimal=1)

#    pairs = MatchMany(imai.PHNC1, imai.Propensity, method = 'knn', k = 5, replace = False)
#    data_matched = whichMatched(pairs, imai, many = True, unique = True)
