"""
A set of rough functions to play around with, that may later be incorporated into a class.
"""

import numpy as np
import itertools

def Kelly(p,a=1.0,b=1.0):
    """
    Function that returns the kelly fraction for a bet of probability p of winning, and probability 1-p of losing.
    Upon losing, a player wins b times the amount bet, and loses a times the amount bet.

    Parameters
    ----------
    p: float
        The probability of winning the bet
    a: float
        The factor the amount bet will multiplied by upon losing
    b: float
        The factor the amount bet will multiplied by upon winning

    Returns
    -------
    The fraction that one one should bet that maximises the log growth rate of bankroll

    """
    m = b*p - a*(1-p)
    return float(m)/float(a)/float(b)
