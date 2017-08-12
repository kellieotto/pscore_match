"""
This module contains a class to estimate propensity scores.
"""

from __future__ import division
import numpy as np
import scipy
from scipy.stats import binom, hypergeom, gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

################################################################################
##################### Base Propensity Score Class ##############################
################################################################################

class PropensityScore(object):
    """
    Estimate the propensity score for each observation.
    
    The compute method uses a generalized linear model to regress treatment on covariates to estimate the propensity score. 
    This is not the only way to estimate the propensity score, but it is the most common.
    The two options allowed are logistic regression and probit regression.
    """

    def __init__(self, treatment, covariates):
        """
        Parameters
        -----------        
        treatment : array-like
            binary treatment assignment
        covariates : pd.DataFrame
            covariates, one row for each observation
        """
        assert treatment.shape[0]==covariates.shape[0], 'Number of observations in \
            treated and covariates doesnt match'
        self.treatment = treatment
        self.covariates = covariates
        
    def compute(self, method='logistic'):
        """
        Compute propensity score and measures of goodness-of-fit
        
        Parameters
        ----------
        method : str
            Propensity score estimation method. Either 'logistic' or 'probit'
        """
        predictors = sm.add_constant(self.covariates, prepend=False)
        if method == 'logistic':
            model = sm.Logit(self.treatment, predictors).fit(disp=False, warn_convergence=True)
        elif method == 'probit':
            model = sm.Probit(self.treatment, predictors).fit(disp=False, warn_convergence=True)
        else:
            raise ValueError('Unrecognized method')
        return model.predict()