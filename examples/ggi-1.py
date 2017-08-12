from pscore_match.data import gerber_green_imai
from pscore_match.pscore import PropensityScore
from pscore_match.match import Match, whichMatched
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

imai = gerber_green_imai()

# Create interaction terms
imai['PERSONS1_VOTE961'] = (imai.PERSONS==1)*imai.VOTE961
imai['PERSONS1_NEW'] = (imai.PERSONS==1)*imai.NEW
treatment = np.array(imai.PHNC1)
cov_list = ['PERSONS', 'VOTE961', 'NEW', 'MAJORPTY', 'AGE', 'WARD', 'AGE2',
                            'PERSONS1_VOTE961', 'PERSONS1_NEW']
covariates = imai[cov_list]
pscore = PropensityScore(treatment, covariates).compute()

pairs = Match(treatment, pscore)
pairs.create(method='many-to-one', many_method='knn', k=5, replace=True)
data_matched = whichMatched(pairs, pd.DataFrame({'pscore': pscore, 'treatment' :treatment, 'voted':imai.VOTED98}))