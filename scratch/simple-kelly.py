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

def kelly_1D(outcomes, p, bankroll):
	

tosses = generate_coin_tosses(p_success = 0.55, n_trials = 10000)
