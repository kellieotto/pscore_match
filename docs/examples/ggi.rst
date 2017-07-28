Gerber-Green-Imai replication
=============================

We reproduce the results from Imai (2005), who replicated and evaluated the field experiment done by Gerber and Green (2000).::

    from pscore_match.data import gerber_green_imai
    from pscore_match.pscore import PropensityScore
    from pscore_match.match import Match

    imai = gerber_green_imai()
    treatment = imai.PHNC1
    cov_list = ['PERSONS', 'VOTE961', 'NEW', 'MAJORPTY', 'AGE', 'WARD', 'AGE2']
    covariates = imai[cov_list]
    pscore = PropensityScore(treatment, covariates).compute()

    pairs = Match(treatment, pscore)
    pairs.create(method='many-to-one', many_method='knn', k=2, replace=False)
