#!/usr/local/bin/env python
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


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

def wealth_afer_n(initial_wealth, n_success, n_failure, kelly_fraction):
	"""
	Returns the amount of wealth after n trials, resulting from betting using Kelly criterion
	Parameters
	----------
	initial_wealth: float
		The initial amount of wealth
	n_success: int
		Number of successes in n trials
	n_failure: int
		Number of failurs in n trials
	kelly_fraction: float
		The value of kelly fraction

	Returns
	-------
	x_n: float
		Amount of wealth after n (where n = n_sucess + n_failure) trials
	"""
	x_n = initial_wealth*((1 + kelly_fraction)**n_success)*((1 - kelly_fraction)**n_failure)
	return x_n

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
    g: float
	    The expectation of the log growth rate of bankroll
	"""
	#FIXME: Decide if we should use log or ln
	#g = ((1-p)*np.log10(1 - a*f)) + (p*np.log10(1 + b*f))
	g = ((1-p)*np.log(1 - a*f)) + (p*np.log(1 + b*f))
	return g

def variance_log_return(p, f, a=1.0, b=1.0):
	"""
	Calculates the variance log return. This is given by:
	s^2 = p*q*{ln[(1 + a*f)/(1 - b*f)]}^2

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
    The variance of the log growth rate of bankroll
	"""
	s_squared = p*(1-p)*((np.log((1 + a*f)/(1 - b*f)))**2)
	return s_squared

def probability_reaching_target(target = 2, n_trials = 10000):
	"""
	Returns the probability of reaching a target wealth on or before n trials, when betting using Kelly criterion

	Parameters
	----------
	target: float
		The value of target wealth
	n: int
		The number of trials on or before which target wealth is reached 
	
	"""
	p = 0.51
	k_f = kelly_fraction(p)
	g = expected_log_return(p, k_f)
	s_squared = variance_log_return(p, k_f)

	T = s_squared*n_trials
	b = np.log(target)
	a = -g/s_squared
	alpha = (-g*(np.sqrt(n_trials)))/np.sqrt(s_squared)
	beta = np.log(target)/(np.sqrt(s_squared)*np.sqrt(n_trials))
	N1 = norm(s_squared, T)
	u = np.sqrt(s_squared) - 2*b
	N2 = norm(u, T)
	
	prob = N1.pdf(-alpha - beta) + (np.exp(-2.0*a*b))*N2.pdf(alpha - beta)
	#print prob
	

def probability_exceeding_target(target = 2, n_trials = 10000):
	"""
	Returns the probability of having wealth euqal to or more than the target amount at the end of n_trials 
	Parameters
	----------
	target: float
		The value of target wealth
	n: int
		The number of trials on or before which target wealth is reached 

	"""

	p = 0.51
	k_f = kelly_fraction(p)
	k_f = 0.0117
	g = expected_log_return(p, k_f)
	s_squared = variance_log_return(p, k_f)

	T = s_squared*n_trials
	b = np.log(target)
	a = -g/s_squared
	alpha = (-g*(np.sqrt(n_trials)))/np.sqrt(s_squared)
	beta = np.log(target)/(np.sqrt(s_squared)*np.sqrt(n_trials))
	N = norm(0.0, T)
	prob = N.pdf(-alpha - beta)
	print prob


def probability_fractional_loss(fraction):
	"""
	Returns the probability of being reduced to a given fraction of the initial wealth
	Parameters
	----------
	fraction: float
		The fraction of initial wealth left after betting
	"""

	p = 0.51
	k_f = kelly_fraction(p)
	g = expected_log_return(p, k_f)
	s_squared = variance_log_return(p, k_f)
	prob = fraction**((2*g)/s_squared)
	print prob

def test_growth_rate(n, p, a, b):
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





def main():
	"""
	Run various test to check these functions
	"""

	n = 10000
	p = 0.55
	a = 1.0
	b = 2.0
	#test_growth_rate(n, p, a, b)
	probability_reaching_target()
	probability_exceeding_target()
	probability_fractional_loss(0.5)

if __name__ == '__main__':
	main()