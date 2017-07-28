"""Standard test data."""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import os as _os
import numpy as np
import pandas as pd
from .. import data_dir

__all__ = ['gerber_green_imai',
           'dehejia_wahba']

def gerber_green_imai():
    """
    This is the dataset from Imai (2005) used to replicate and evaluate
    the field experiment done by Gerber and Green (2000).
    
    Notes
    -----
    .. Gerber, Alan S. and Donald P. Green. 2000. "The effects of canvassing,
    telephone calls, and direct mail on voter turnout: a field experiment."
    American Political Science Review 94: 653-663.
    
    .. Gerber, Alan S. and Donald P. Green. 2005. "Correction to Gerber and Green (2000),
    replication of disputed findings, and reply to Imai (2005)." American Political 
    Science Review 99: 301-313.
    
    .. Imai, Kosuke. 2005. "Do get-out-the-vote calls reduce turnout? The importance of 
    statistical methods for field experiments." American Political Science Review 99: 
    283-300.
    """
    fin = _os.path.join(data_dir, 'GerberGreenImai.txt')
    data = pd.read_table(fin, sep = '\s+')
    data.index = range(data.shape[0])
    return data

def dehejia_wahba():
    """
    Data from Dehejia and Wahba (1999, 2002) used to replicate and evaluate the matching
    results of Lalonde (1986).

    .. Dehejia, Rajeev and Sadek Wahba. 1999. "Causal effects in non-experimental studies: 
    Reevaluating the evaluation of training programs." Journal of the American Statistical
    Association 94 (448): 1053-1062.

    .. Dehejia, Rajeev and Sadek Wahba. 2002. "Propensity score matching methods for non-
    experimental causal studies." Review of Economics and Statistics 84: 151-161.
    
    .. LaLonde, Robert. 1986. "Evaluating the econometric evaluations of training programs 
    with experimental data." American Economic Review 76 (4): 604-620.
    """
    names = ['Treated', 'Age', 'Education', 'Black', 'Hispanic', 'Married',
             'Nodegree', 'RE74', 'RE75', 'RE78']
    fin_tr = _os.path.join(data_dir, 'nswre74_treated.txt')
    fin_ct = _os.path.join(data_dir, 'nswre74_control.txt')
    treated = pd.read_table(fin_tr, sep = '\s+',
                            header = None, names = names)
    control = pd.read_table(fin_ct, sep='\s+', 
                            header = None, names = names)
    data = pd.concat([treated, control])
    data.index = range(data.shape[0])
    return data