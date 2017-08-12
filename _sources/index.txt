Welcome to pscore_match's documentation!
========================================

`pscore_match` is a package for analysis of observational data using propensity score matching.

`Download the package here!`__

.. __: https://github.com/kellieotto/pscore_match

Questions, comments, and contributions are welcome.

About
=====

Propensity score matching is a method to match treated and control group individuals in observational studies in order to better estimate the effect of the treatment or exposure on the outcome of interest.  
This is done in two steps:
1) Estimate the **propensity score**, the conditional probability of receiving treatment given the covariates
2) Match pairs with the same (or similar) propensity scores.  

The idea is that the individuals in these pairs were equally likely to receive the treatment, so their outcome should be unconfounded with the covariates.
The math is a bit more complicated than that, but that's the general intuition behind the method.
Pairs should be similar in their covariates and outcomes, except for the effect of the treatment.

The tricky thing about propensity score matching is that there's no single good way to do it.
There are different methods which give different results, but there's one testable implication:
the best estimate of the propensity score is the one which gives you the best covariate balance.
This package includes functions to test for covariate balance.
Typically, you would iterate between estimating propensity scores and checking for covariate balance.
If the balance is insufficient, you'd redo the estimation using different covariates.
 
If your estimated propensities are wrong, then you're screwed right off the bat.
But assuming they're alright, then how do you pick the "best" treatment-control pairs?
You could try every possible pairing and minimize the within-pair differences in propensity, but that's computationally intensive.
What's typically done is greedy matching, but even then there are a number of factors to decide: in what order do we match the treated to controls?
Do we match with or without replacement, allowing one control to be matched to one or more cases?
Do we use a caliper to set a maximum difference in propensities, and if so how do we pick the caliper?
The package includes options to vary all of these choices.

API Reference
=============
Contents:

.. toctree::
   :maxdepth: 2

   api/index.rst
   examples/index.rst
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

