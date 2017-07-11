"""
Implements several propensity score estimation, balance diagnostics for 
group characteristics, average treatment effect on the treated (ATT) estimates, 
and bootstraping to estimate standard errors of the estimated ATT.
"""

from __future__ import division
import numpy as np
import scipy
from scipy.stats import binom, hypergeom, gaussian_kde
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from ModelMatch import binByQuantiles
import statsmodels.api as sm

################################################################################
##################### Base Propensity Score Class ##############################
################################################################################

def computePropensityScore(formula, data, verbosity=1):
    '''
    Compute propensity scores 
    
    Parameters
    -----------
    formula : string 
        Must have the form 'Treatment~ covariate1 + covariate2 + ...', where these are column names in data
    data : array-like 
        Matrix with columns corresponding to terms in the formula
    verbosity : bool
        whether or not to print glm summary
    
    dependencies: LogisticRegression from sklearn.linear_model
                  statsmodels as sm
    '''
        
    ####### Using LogisticRegression from sklearn.linear_model    
    #propensity = LogisticRegression()
    #propensity.fit(predictors, groups)
    #return propensity.predict_proba(predictors)[:,1]
    
    ####### Using sm.GLM
    #predictors = sm.add_constant(predictors, prepend=False)
    #glm_binom = sm.GLM(groups, predictors, family=sm.families.Binomial())

    ####### Using sm.formula.glm with formula call
    glm_binom = sm.formula.glm(formula = formula, data = data, family = sm.families.Binomial())
    res = glm_binom.fit()
    if verbosity:
        print res.summary()
    return res.fittedvalues



################################################################################
############################## estimation funcs ################################
################################################################################

def averageTreatmentEffect(groups, response, matches):
    '''
    Computes ATT using difference in means.
    The data passed in should already have unmatched individuals and duplicates removed.

    Parameters
    -----------
    groups : pd.Series 
        treatment assignment. Must be 2 groups
    response : pd.Series 
        response measurements. Indices should match those of groups.
    matches : 
        output of Match or MatchMany
    '''
    if len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')
    
    groups = (groups == groups.unique()[0])
    response = response.groupby(response.index).first()
    response1 = []; response0 = []
    for k in matches.keys():
        response1.append(response[k])
        response0.append( (response[matches[k]]).mean() )   # Take mean response of controls matched to treated individual k
    return np.array(response1).mean() - np.array(response0).mean()


def regressAverageTreatmentEffect(groups, response, covariates, matches=None, verbosity = 0):
    '''
    Computes ATT by regression.
    This works for one-to-one matching.   The data passed in should already have unmatched individuals removed.
    Weights argument will be added later for one-many matching
    
    Parameters
    -----------
    groups : pd.Series 
        treatment assignment. Must be 2 groups
    response : pd.Series 
        response measurements. Indices should match those of groups.
    covariates : pd.DataFrame 
        the covariates to include in the linear regression
    matches : 
        output of Match or MatchMany
    
    Dependencies: statsmodels.api as sm, pandas as pd
    '''
    if len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')
    
    weights = pd.Series(data = np.ones(len(groups)), index = groups.index)
    if matches:
        ctrl = [m for matchset in matches.values() for m in matchset]    
        matchcounts = pd.Series(ctrl).value_counts()
        for i in matchcounts.index:
            weights[i] = matchcounts[i]
        if verbosity:
            print weights.value_counts(), weights.shape
    X = pd.concat([groups, covariates], axis=1)
    X = sm.add_constant(X, prepend=False)
    linmodel = sm.WLS(response, X, weights = weights).fit()
    return linmodel.params[0], linmodel.bse[0]

def sampleWithinGroups(groups, data):
    '''
    To use in bootstrapping functions.  Sample with replacement from each group, return bootstrap sample dataframe.
    
    Parameters
    -----------
    groups : pd.Series 
        treatment assignment. Must be 2 groups
    data : Dataframe 
        observations from which to create bootstrap sample.
    '''
    bootdata = pd.DataFrame()
    for g in groups.unique():
        sample = np.random.choice(data.index[data.groups==g], sum(groups == g), replace = True)
        newdata =(data[data.groups==g]).ix[sample]
        bootdata = bootdata.append(newdata)
    bootdata.index = range(len(groups))
    return bootdata


def bootstrapATT(groups, response, propensity, many=True, B = 500, method = "caliper", k = 1, caliper = 0.05, caliper_method = "propensity", replace = False):
    '''
    Computes bootstrap standard error of the average treatment effect on the treated.
    Sample observations with replacement, within each treatment group. Then match them and compute ATT.
    Repeat B times and take standard deviation.
    
    Parameters
    -----------
    groups : pd.Series 
        treatment assignment. Must be 2 groups
    response : pd.Series 
        response measurements
    propensity : pd.Series 
        propensity scores
    many : bool
        are we using one-many matching?
    B : int
        number of bootstrap replicates. Default is 500
    caliper, caliper_method, replace = arguments to pass to Match or MatchMany
    method, k = arguments to pass to MatchMany
    '''
    if len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')

    data = pd.DataFrame({'groups':groups, 'response':response, 'propensity':propensity})
    boot_ate = np.empty(B)
    for i in range(B):
        bootdata = sampleWithinGroups(groups, data)
        if many:
            pairs = MatchMany(bootdata.groups, bootdata.propensity, method = method, k = k, caliper = caliper, caliper_method = caliper_method, replace = replace)
        else:
            pairs = Match(bootdata.groups, bootdata.propensity, caliper = caliper, caliper_method = caliper_method, replace = replace)
        boot_ate[i] = averageTreatmentEffect(bootdata.groups, bootdata.response, matches = pairs)
    return boot_ate.std()


def bootstrapRegression(groups, response, propensity, covariates, many = True, B = 500, method = "caliper", k = 1, caliper = 0.05, caliper_method = "propensity", replace = False):
    '''
    Computes bootstrap standard error of the ATT.
    Sample observations with replacement, within each treatment group. Then match them and compute ATT.
    Repeat B times and take standard deviation.
    
    Parameters
    -----------
    groups : pd.Series 
        treatment assignment. Must be 2 groups
    response : pd.Series 
        response measurements
    propensity : pd.Series 
        propensity scores
    covariates : pd.DataFrame 
        covariates to use in regression
    many : bool
        are we using one-many matching?
    B : int
        number of bootstrap replicates. Default is 500
    caliper, caliper_method, replace = arguments to pass to Match or MatchMany
    method, k = arguments to pass to MatchMany
    '''
    if len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')

    data = pd.DataFrame({'groups':groups, 'response':response, 'propensity':propensity})
    data = pd.concat([data, covariates], axis=1)
    boot_ate = np.empty(B)
    for i in range(B):
        bootdata = sampleWithinGroups(groups, data)
        if many:
            pairs = MatchMany(bootdata.groups, bootdata.propensity, method = method, k = k, caliper = caliper, caliper_method = caliper_method, replace = replace)
            matched = whichMatched(pairs, bootdata, many = True, unique = True)
            boot_ate[i] = regressAverageTreatmentEffect(matched.groups, matched.response, matched.ix[:,3:], matches=pairs, verbosity = 0)[0]
        else:
            pairs = Match(bootdata.groups, bootdata.propensity, caliper = caliper, caliper_method = caliper_method, replace = replace)
            matched = whichMatched(pairs, bootdata, many = False)
            boot_ate[i] = regressAverageTreatmentEffect(matched.groups, matched.response, matched.ix[:,3:], matches=None, verbosity = 0)[0]
    return boot_ate.std()
    

def Balance(groups, covariates):
    '''
    Computes absolute difference of means and standard error for covariates by group
    '''
    means = covariates.groupby(groups).mean()
    dist = abs(means.diff()).ix[1]
    std = covariates.groupby(groups).std()
    n = groups.value_counts()
    se = std.apply(lambda(s): np.sqrt(s[0]**2/n[0] + s[1]**2/n[1]))
    return dist, se

def plotScores(groups, propensity, matches, many=True):
    '''
    Plot density of propensity scores for each group before and after matching
    
    Parameters
    ----------- 
    groups : pd.Series
        treatment assignment, pre-matching
    propensity : pd.Series
        propensity scores, pre-matching
    matches :
        output of Match or MatchMany
    many : bool
        True if one-many matching was done (default is True), otherwise False
    '''
    pre = pd.DataFrame({'groups':groups, 'propensity':propensity})    
    post = whichMatched(matches, pre, many = many, unique = False)
    
    plt.figure(1)
    plt.subplot(121)
    density0 = scipy.stats.gaussian_kde(pre.propensity[pre.groups==0])
    density1 = scipy.stats.gaussian_kde(pre.propensity[pre.groups==1])
    xs = np.linspace(0,1,1000)
    plt.plot(xs,density0(xs),color='black')
    plt.fill_between(xs,density1(xs),color='gray')
    plt.title('Before Matching')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    
    plt.subplot(122)
    density0_post = scipy.stats.gaussian_kde(post.propensity[post.groups==0])
    density1_post = scipy.stats.gaussian_kde(post.propensity[post.groups==1])
    xs = np.linspace(0,1,1000)
    plt.plot(xs,density0_post(xs),color='black')
    plt.fill_between(xs,density1_post(xs),color='gray')
    plt.title('After Matching')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.show()
    

################################################################################
############################## some sample code ################################
################################################################################

