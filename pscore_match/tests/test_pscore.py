from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

from nose.plugins.attrib import attr
from nose.tools import assert_raises, raises
import numpy as np
import pandas as pd

from ..pscore import PropensityScore

def test_imai_data():
    imai = pd.read_table('../data/GerberGreenImai.txt', sep = '\s+')
    treatment = imai.PHNC1
    cov_list = ['PERSONS', 'VOTE961', 'NEW',
                'MAJORPTY', 'AGE', 'WARD', 'AGE2']
    covariates = imai[cov_list]
    #form = 'PHNC1 ~ PERSONS + VOTE961 + NEW + MAJORPTY + AGE + \
    #                    WARD + PERSONS:VOTE961 + PERSONS:NEW + AGE2'
    pscore = PropensityScore(treatment, covariates).compute()
    np.testing.assert_equal(np.mean(pscore[treatment==1]), 0.030815395724674208)
    np.testing.assert_equal(np.mean(pscore[treatment==0]), 0.022622245062938591)

#    pairs = MatchMany(imai.PHNC1, imai.Propensity, method = 'knn', k = 5, replace = False)
#    data_matched = whichMatched(pairs, imai, many = True, unique = True)
