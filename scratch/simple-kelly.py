#!/usr/local/bin/env python
import random
import numpy as np
import matplotlib.pyplot as plt


def generate_coin_tosses(p_success = 0.5, n_trials = 100):
	"""
	Generates a sequence of coin tosses, with heads distributed according to
	a specified success probability.
	
	Parameters
	----------
	p_success : float
		Probability of obtaining head on a coin flip
	n_trials : int
		Number of coin flips

	Returns
	-------
	outcomes : list
		A list of length n_trails, where each element of this list if either H or T
	"""
	
	outcomes = []
	for i in xrange(n_trials):
		r = random.uniform(0, 1)
		if r < p_success:
			outcomes.append("H")
			#heads += 1
		else:
			outcomes.append("T")
	return outcomes

"""
A set of rough functions to play around with, that may later be incorporated into a class.
"""

def kelly_fraction(p,a=1.0,b=1.0):
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

def expected_log_return(p, f, a=1.0, b=1.0):
	"""
	Calculates expected log return per trial (g). This is given by:
	g = q*log(1 - a*f) + p*log(1 + b*f)

	Parameters
	----------
    p: float
        The probability of winning the bet
    a: float
        The factor the amount bet will multiplied by upon losing
    b: float
        The factor the amount bet will multiplied by upon winning
    f: float
    	Fraction of bankroll to bet

    Returns
    -------
    The expectation of the log growth rate of bankroll
	"""

	g = ((1-p)*np.log10(1 - a*f)) + (p*np.log10(1 + b*f))

	return g

n = 10000
p = 0.55
a = 1.0
b = 2.0

k_f = kelly_fraction(p, a, b)
g = expected_log_return(p, k_f, a, b)
print "Calculated Kelly fraction: ", k_f
print "Calculated expected growth rate: ", g

test_fractions = np.arange(0.0, 1.0, 0.025)
test_rates = [expected_log_return(p, i, a, b) for i in test_fractions]

test_kelly = test_fractions[test_rates.index(max(test_rates))]
print "Estimated fraction that maximized log growth rate: ", test_kelly
plt.plot(test_fractions, test_rates, 'o')
#plt.show()
plt.savefig("test_g.png")