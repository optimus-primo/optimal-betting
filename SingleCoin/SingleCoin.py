import numpy as np
#import numpy.testing as npt
from scipy.stats import norm

class SingleCoinBetting(object):
    """
    TODO: tidy up this
    Class to sample and predict the optimal kelly betting fraction for repeated betting.

    The game is fixed: i.e. the game consists of probability of winning, and the payoffs. The purpose of this class is
    to find out the properties of the various betting strategies (fractions) that one can make.

    Some notation based on Thorpe Chapter...
    """
    def __init__(self,p=0.6, a=1.0, b=1.0, initial_wealth=100.0):
        """
        Initialize class.

        Parameters
        ----------
        p: float
            the probability of drawing heads and winning in a coin toss
        initial_wealth: float
            the initial wealth of the player
        a: float
            The factor the amount bet will multiplied by upon losing
        b: float
            The factor the amount bet will multiplied by upon winning
        """
        self.p = float(p)
        self.initial_wealth = float(initial_wealth)
        self.a = float(a)
        self.b = float(b)

        # Find the optimal Kelly betting fraction for this coin, and it's various growth properties
        self.f_kelly = self.kelly_fraction()
        self.g = self.expected_log_return()
        self.g_var = self.variance_log_return()

    def kelly_fraction(self):
        """
        Function that returns the kelly fraction for a bet of probability p of winning, and probability 1-p of losing.
        Upon losing, a player wins b times the amount bet, and loses a times the amount bet.

        Returns
        -------
        The fraction that one one should bet that maximises the log growth rate of bankroll
        """
        m = self.b * self.p - self.a * (1 - self.p)
        return m / self.a / self.b

    def max_fraction(self):
        """
        Calculates the maximum one should bet before the fraction will lead to ruin

        Return
        ------
        """
        pass

    def expected_log_return(self, f=None):
        """
        Calculates expected log return per trial (g). This is given by:

        g = q*log(1 - a*f) + p*log(1 + b*f)

        Parameter
        ---------
        f: float
            Fraction of bankroll to bet

        Returns
        -------
        g: float
            The expectation of the log growth rate of bankroll
        """
        if f is None:
            f = self.f_kelly

        # FIXME: Decide if we should use log or ln
        # g = ((1-p)*np.log10(1 - a*f)) + (p*np.log10(1 + b*f))
        g = ((1 - self.p) * np.log(1 - self.a * f)) + (self.p * np.log(1 + self.b * f))
        return g

    def variance_log_return(self, p=None, f=None, a=None, b=None):
        """
        Calculates the variance log return. This is given by:
        s^2 = p*q*{ln[(1 + a*f)/(1 - b*f)]}^2

        Parameter
        ---------
        f: float
        	Fraction of bankroll to bet

        Returns
        -------
        The variance of the log growth rate of bankroll
        """

        if f is None:
            f = self.f_kelly

        s_squared = self.p * (1 - self.p) * ((np.log((1 + self.a * f) / (1 - self.b * f))) ** 2)
        return s_squared

    def prob_reaching_target(self,target, n_trials, f=None):
        """
        Returns the probability of reaching a target wealth on or before n trials, when betting using Kelly criterion

        Parameters
        ----------s
        target: float
            The value of target wealth
        n: int
            The number of trials on or before which target wealth is reached
        f: float
            The fraction of ones wealth that will be bet at each round

        Returns
        -------
        prob: float
            The stated probability
        """
        if f is None:
            g = self.g
            g_var = self.g_var
        else:
            g = self.expected_log_return(f)
            g_var = self.variance_log_return(f)

        #TODO: account for initial wealth
        alpha = (-g * (np.sqrt(n_trials))) / np.sqrt(g_var)
        beta = np.log(target) / (np.sqrt(g_var) * np.sqrt(n_trials))
        N = norm(0.0, 1.0)

        prob = (1 - N.cdf(alpha + beta)) + (np.exp(-2.0 * a * b)) * N.cdf(alpha - beta)
        return prob

    def probability_exceeding_target(self, target, n_trials, f=None):
        """
        Returns the probability of having wealth equal to or more than the target amount at the end of n_trials

        Parameters
        ----------
        p: float
            The probability of winning the bet
        target: float
            The value of target wealth
        n: int
        The number of trials on or before which target wealth is reached

        Returns
        -------
        prob: float
            The stated probability
        """
        if f is None:
            g = self.g
            g_var = self.g_var
        else:
            g = self.expected_log_return(f)
            g_var = self.variance_log_return(f)


        #TODO: account for initial wealth
        alpha = (-g * (np.sqrt(n_trials))) / np.sqrt(g_var)
        beta = np.log(target) / (np.sqrt(g_var) * np.sqrt(n_trials))
        N = norm(0.0, 1.0)

        prob = 1 - N.cdf(alpha + beta)
        return prob

    def probability_fractional_loss(self,f=None):
        """
        Returns the probability of being reduced to a given fraction of the initial wealth

        Parameters
        ----------
        p: float
            The probability of winning the bet
        fraction: float
            The fraction of initial wealth left after betting

        Returns
        -------
        prob: float
            The stated probability
        """
        if f is None:
            f = self.f_kelly
            g = self.g
            g_var = self.g_var
        else:
            g = self.expected_log_return(f)
            g_var = self.variance_log_return(f)

        #TODO: account for initial wealth
        prob = f ** ((2 * g) / g_var)

        return prob

    def wealth_afer_n(initial_wealth, n_success, n_failure, kelly_fraction):
        pass

        # Generator function?