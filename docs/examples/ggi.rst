Gerber-Green-Imai replication
=============================

We reproduce the results from Imai (2005), who replicated and evaluated the field experiment done by Gerber and Green (2000).

.. code::
    imai = gerber_green_imai()
    treatment = imai.PHNC1
    cov_list = ['PERSONS', 'VOTE961', 'NEW',
                'MAJORPTY', 'AGE', 'WARD', 'AGE2']
    covariates = imai[cov_list]
    pscore = PropensityScore(treatment, covariates).compute()
	
	pairs = Match(imai.PHNC1, pscore)
	pairs.create(method='many-to-one', many_method='knn', k=25, replace=False)

.. plot::
	plt.figure(1)