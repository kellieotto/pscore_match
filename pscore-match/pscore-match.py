"""
Implements several types of propensity-score matching, balance diagnostics for 
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
    
    Inputs:
    formula = string of the form 'Treatment~ covariate1 + covariate2 + ...', where these are column names in data
    data = matrix-like object with columns corresponding to terms in the formula
    verbosity = whether or not to print glm summary
    
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
############################# Base Matching Class ##############################
################################################################################

def Match(groups, propensity, caliper = 0.05, caliper_method = "propensity", replace = False):
    ''' 
    Implements greedy one-to-one matching on propensity scores.
    
    Inputs:
    groups = Array-like object of treatment assignments.  Must be 2 groups
    propensity = Array-like object containing propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    caliper = a numeric value, specifies maximum distance (difference in propensity scores or SD of logit propensity) 
    caliper_method = a string: "propensity" (default) if caliper is a maximum difference in propensity scores,
            "logit" if caliper is a maximum SD of logit propensity, or "none" for no caliper
    replace = Logical for whether individuals from the larger group should be allowed to match multiple individuals in the smaller group.
        (default is False)
    
    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match.
    '''

    # Check inputs
    if any(propensity <=0) or any(propensity >=1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not(0<=caliper<1):
        if caliper_method == "propensity" and caliper>1:
            raise ValueError('Caliper for "propensity" method must be between 0 and 1')
        elif caliper<0:
            raise ValueError('Caliper cannot be negative')
    elif len(groups)!= len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')
        
    
    # Transform the propensity scores and caliper when caliper_method is "logit" or "none"
    if caliper_method == "logit":
        propensity = log(propensity/(1-propensity))
        caliper = caliper*np.std(propensity)
    elif caliper_method == "none":
        caliper = 0
    
    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups[groups == 1].index; N2 = groups[groups == 0].index
    g1, g2 = propensity[groups == 1], propensity[groups == 0]
    # Check if treatment groups got flipped - the smaller should correspond to N1/g1
    if len(N1) > len(N2):
       N1, N2, g1, g2 = N2, N1, g2, g1
        
        
    # Randomly permute the smaller group to get order for matching
    morder = np.random.permutation(N1)
    matches = {}

    
    for m in morder:
        dist = abs(g1[m] - g2)
        if (dist.min() <= caliper) or not caliper:
            matches[m] = dist.argmin()    # Potential problem: check for ties
            if not replace:
                g2 = g2.drop(matches[m])
    return (matches)



def MatchMany(groups, propensity, method = "caliper", k = 1, caliper = 0.05, caliper_method = "propensity", replace = True):
    ''' 
    Implements greedy one-to-many matching on propensity scores.
    
    Inputs:
    groups = Array-like object of treatment assignments.  Must be 2 groups
    propensity = Array-like object containing propensity scores for each observation. Propensity and groups should be in the same order (matching indices)
    method = a string: "caliper" (default) to select all matches within a given range, "knn" for k nearest neighbors,
    k = an integer (default is 1). If method is "knn", this specifies the k in k nearest neighbors
    caliper = a numeric value, specifies maximum distance (difference in propensity scores or SD of logit propensity) 
    caliper_method = a string: "propensity" (default) if caliper is a maximum difference in propensity scores,
            "logit" if caliper is a maximum SD of logit propensity, or "none" for no caliper
    replace = Logical for whether individuals from the larger group should be allowed to match multiple individuals in the smaller group.
        (default is True)
    
    Output:
    A series containing the individuals in the control group matched to the treatment group.
    Note that with caliper matching, not every treated individual may have a match within calipers.
        In that case we match it to its single nearest neighbor.  The alternative is to throw out individuals with no matches, but then we'd no longer be estimating the ATT.
    '''

    # Check inputs
    if any(propensity <=0) or any(propensity >=1):
        raise ValueError('Propensity scores must be between 0 and 1')
    elif not(0<=caliper<1):
        if caliper_method == "propensity" and caliper>1:
            raise ValueError('Caliper for "propensity" method must be between 0 and 1')
        elif caliper<0:
            raise ValueError('Caliper cannot be negative')
    elif len(groups)!= len(propensity):
        raise ValueError('groups and propensity scores must be same dimension')
    elif len(groups.unique()) != 2:
        raise ValueError('wrong number of groups: expected 2')
        
    
    # Transform the propensity scores and caliper when caliper_method is "logit" or "none"
    if method == "caliper":
        if caliper_method == "logit":
            propensity = log(propensity/(1-propensity))
            caliper = caliper*np.std(propensity)
        elif caliper_method == "none":
            caliper = 0
    
    # Code groups as 0 and 1
    groups = groups == groups.unique()[0]
    N = len(groups)
    N1 = groups[groups == 1].index; N2 = groups[groups == 0].index
    g1, g2 = propensity[groups == 1], propensity[groups == 0]
    # Check if treatment groups got flipped - the smaller should correspond to N1/g1
    if len(N1) > len(N2):
       N1, N2, g1, g2 = N2, N1, g2, g1
        
        
    # Randomly permute the smaller group to get order for matching
    morder = np.random.permutation(N1)
    matches = {}
    
    for m in morder:
        dist = abs(g1[m] - g2)
        dist.sort()
        if method == "knn":
            caliper = dist.iloc[k-1]
        # PROBLEM: when there are ties in the knn. 
        # Need to randomly select among the observations tied for the farthest eacceptable distance
        keep = np.array(dist[dist<=caliper].index)
        if len(keep):
            matches[m] = keep
        else:
            matches[m] = [dist.argmin()]
        if not replace:
            g2 = g2.drop(matches[m])
    return (matches)


    
def whichMatched(matches, data, many = False, unique = False):
    ''' 
    Simple function to convert output of Matches to DataFrame of all matched observations
    Inputs:
    matches = output of Match
    data = DataFrame of covariates
    many = Boolean indicating if matching method is one-to-one or one-to-many
    unique = Boolean indicating if duplicated individuals (ie controls matched to more than one case) should be removed
    '''

    tr = matches.keys()
    if many:
        ctrl = [m for matchset in matches.values() for m in matchset]
    else:
        ctrl = matches.values()
    # need to remove duplicate rows, which may occur in matching with replacement
    temp = pd.concat([data.ix[tr], data.ix[ctrl]])
    if unique == True:
        return temp.groupby(temp.index).first()
    else:
        return temp
        

def getWeights(matches, groups):
    ''' computes weights for mean & regression according to how many times a control was matched in one-many matching'''
    
    ctrl = [m for matchset in matches.values() for m in matchset]
    weights = groups.copy()
    for c in ctrl:
        weights[c] += 1
    return weights
    
    
def whichMatched(matches, data, many = False, unique = False):
    ''' 
    Simple function to convert output of Matches to DataFrame of all matched observations
    Inputs:
    matches = output of Match
    data = DataFrame of covariates
    many = Boolean indicating if matching method is one-to-one or one-to-many
    unique = Boolean indicating if duplicated individuals (ie controls matched to more than one case) should be removed
    '''

    tr = matches.keys()
    if many:
        ctrl = [m for matchset in matches.values() for m in matchset]
    else:
        ctrl = matches.values()
    # need to remove duplicate rows, which may occur in matching with replacement
    temp = pd.concat([data.ix[tr], data.ix[ctrl]])
    if unique == True:
        return temp.groupby(temp.index).first()
    else:
        return temp
        

################################################################################
############################## estimation funcs ################################
################################################################################

def averageTreatmentEffect(groups, response, matches):
    '''
    Computes ATT using difference in means.
    The data passed in should already have unmatched individuals and duplicates removed.

    Inputs:
    groups = Series containing treatment assignment. Must be 2 groups
    response = Series containing response measurements. Indices should match those of groups.
    matches = output of Match or MatchMany
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
    
    Inputs:
    groups = Series containing treatment assignment. Must be 2 groups
    response = Series containing response measurements. Indices should match those of groups.
    covariates = DataFrame containing the covariates to include in the linear regression
    matches = optional: if using one-many matching, should be the output of MatchMany.
            Use None for one-one matching.
    
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
    To use in bootstrapping functions.  
    Sample with replacement from each group, return bootstrap sample dataframe
    
    Inputs:
    groups = Series containing treatment assignment. Must be 2 groups
    data   = Dataframe containing observations from which to create bootstrap sample.
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
    
    Inputs:
    groups = Series containing treatment assignment. Must be 2 groups
    response = Series containing response measurements
    propensity = Series containing propensity scores
    many = Boolean: are we using one-many matching?
    B = number of bootstrap replicates. Default is 500
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
    
    Inputs:
    groups = Series containing treatment assignment. Must be 2 groups
    response = Series containing response measurements
    propensity = Series containing propensity scores
    covariates = DataFrame containing covariates to use in regression
    many = Boolean: are we using one-many matching?
    B = number of bootstrap replicates. Default is 500
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
    
    Inputs: groups = treatment assignment, pre-matching
            propensity = propensity scores, pre-matching
            matches = output of Match or MatchMany
            many = indicator - True if one-many matching was done (default is True), otherwise False
    '''
    pre = pd.DataFrame({'groups':groups, 'propensity':propensity})    
    post = whichMatched(matches, pre, many = many, unique = False)
    
    plt.figure(1)
    plt.subplot(121)
    density0 = scipy.stats.gaussian_kde(pre.propensity[pre.groups==0])
    density1 = scipy.stats.gaussian_kde(pre.propensity[pre.groups==1])
    xs = np.linspace(0,1,1000)
    #density0.covariance_factor = lambda : 0.5
    #density0._compute_covariance()
    #density1.covariance_factor = lambda : 0.5
    #density1._compute_covariance()
    plt.plot(xs,density0(xs),color='black')
    plt.fill_between(xs,density1(xs),color='gray')
    plt.title('Before Matching')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    
    plt.subplot(122)
    density0_post = scipy.stats.gaussian_kde(post.propensity[post.groups==0])
    density1_post = scipy.stats.gaussian_kde(post.propensity[post.groups==1])
    xs = np.linspace(0,1,1000)
    #density0.covariance_factor = lambda : 0.5
    #density0._compute_covariance()
    #density1.covariance_factor = lambda : 0.5
    #density1._compute_covariance()
    plt.plot(xs,density0_post(xs),color='black')
    plt.fill_between(xs,density1_post(xs),color='gray')
    plt.title('After Matching')
    plt.xlabel('Propensity Score')
    plt.ylabel('Density')
    plt.show()
    

################################################################################
############################## some sample code ################################
################################################################################

