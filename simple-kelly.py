#!/usr/local/bin/env python
import random

def generate_coin_tosses(p_success = 0.5, n_trials = 100):
	"""Generate a sequence of coin tosses, with heads distributed according ti
	specified success probability.
	
	Parameters
	----------
	p_success : float
		Probability of obtaining head on a coin flip
	n_trials : int
		Number of coin flips

	Returns
	-------
	outcomes : list
		A list of length n_trais, where each element of this list if either H or T

	"""
	
	outcomes = []
	for i in xrange(n_trials):
		r = random.uniform(0, 1)
		if r < p_success:
			outcomes.append("H")
			heads += 1
		else:
			outcomes.append("T")
	return outcomes

#def kelly_1D(outcomes, p, bankroll):
"""
A set of rough functions to play around with, that may later be incorporated into a class.
"""

def kelly(p,a=1.0,b=1.0):
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

tosses = generate_coin_tosses(p_success = 0.55, n_trials = 10000)
