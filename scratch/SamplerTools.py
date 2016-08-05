import numpy as np
import itertools

class CoinFlipSampler(object):
    """
    Class to sample simultaneous coin flipping events (Bernoulli processes) where coins can be arbitrarily correlated.
    The correlation between coins is encompassed by the joint distribution over all outcomes. An individual coin flip
    returns either -1 or 1.

    Examples
    --------
    If ncoins = 2, there are 4 possible outcomes: (-1,1),(-1,-1),(1,-1),(1,1). The probability for each
    outcome is specified by the user or is uniformly generated.

    flipper = CoinFlipSampler(2)        # Initialize the class with 2 coins
    print flipper.states             # View all possible outcomes
    print flipper.probs                 # View the randomly generated probability distribution over the outcomes
    flipper.sample_flips(10)            # Generate 10 simultaneous flips of both coins

    """
    def __init__(self,ncoins,probs=None):
        """
        Initialize class to sample simultaneous coin flips whose outcomes are either -1 or 1.

        Parameters
        ----------
        ncoins: int
            The number of coins that will be flipped simultaneously
        probs: list of floats or integers
            Either the probabilities for each of the simultaneous outcomes or the un-normalized weights for each event.
            The length of the list must be equal to 2**ncoins. If left blank, probabilities will be uniformly generated.
        """
        self.ncoins = ncoins
        self.states = self._gen_statespace()

        if probs == None:
            self.probs = self._sample_simplex()
        else:
            if len(probs) != 2**ncoins:
                raise Exception('The length of the probability vector "probs" must equal the size of the state-space =  2**ncoins')
            else:
                probs = np.array(probs)*1.0
                self.probs = probs/np.sum(probs)

    def _gen_statespace(self):
        """
        Enumerates and returns a list of possible outcomes for each simultaneous coin flips.

        Returns
        -------
        list of length ncoins**2
            All possible simultaneous results for coin flips, represented in terms of 1 and -1
        """

        return list(itertools.product([-1,1], repeat=self.ncoins))


    def _sample_simplex(self):
        """
        Sample uniformly from a simplex of a given number of dimensions. Used for sampling the categorical probability mass
        functions.

        Returns
        -------
        np.array of length ndims
            A total of ndims random numbers that sum to 1.
        """
        ndims = 2**self.ncoins
        if ndims <=1:
            raise Exception("ncoins must be greater than or to 2")
        x = np.random.uniform(size=ndims-1)
        x = np.sort(x)
        x = np.hstack((0,x,1))
        return np.diff(x)

    def sample_flips(self,nsamps):
        """
        Sample from state space of simultaneous coin flips using the probabilities provided.

        Parameters
        ----------
        nsamps: int
            The number of samples one wishes to generate

        Returns
        -------
        list of outcomes
            Each outcome is one instance of simultaneous flipping all ncoins, with coin faces represented as -1 or +1.
        """
        outcomes = []
        for flip in range(nsamps):
            rand_ind = np.random.choice(2**self.ncoins,p=self.probs)
            outcomes.append(self.states[rand_ind])
        return outcomes
