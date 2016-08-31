import numpy as np
#import numpy.testing as npt
from scipy.stats import norm

class SingleCoinBetting(object):
    """
    Class to sample and predict the optimal kelly betting fraction for repeated betting.

    The game is fixed, and consists of probability of winning, and the payoffs. The purpose of this class is
    to find out the properties of the various betting strategies (fractions) that one can make.

    Some notation based on Thorpe Chapter...
    """
    def __init__(self,p=0.6, a=1.0, b=1.0, initial_logwealth = 0.0):
        """
        Initialize class.

        Parameters
        ----------
        p: float
            the probability of drawing heads and winning in a coin toss
        initial_logwealth: float
            the log of the initial wealth of the player
        a: float
            The factor the amount bet will multiplied by upon losing
        b: float
            The factor the amount bet will multiplied by upon winning
        """
        self.p = float(p)
        self.logwealth = float(initial_logwealth)
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

    def expected_log_return(self, f=None):
        """
        Calculates expected log return per trial (g). This is given by:

        g = q*log(1 - a*f) + p*log(1 + b*f)

        Parameter
        ---------
        f: float
            Fraction of wealth that will be bet

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

    def variance_log_return(self, f=None):
        """
        Calculates the variance log return. This is given by:
        s^2 = p*q*{ln[(1 + a*f)/(1 - b*f)]}^2

        Parameter
        ---------
        f: float
        	the fraction that will be bet

        Returns
        -------
        The variance of the log growth rate of bankroll
        """

        if f is None:
            f = self.f_kelly

        g_var = self.p * (1 - self.p) * ((np.log((1 + self.a * f) / (1 - self.b * f))) ** 2)
        return g_var

    def prob_reaching_target(self,target, n_trials, f=None):
        """
        Returns the probability of reaching a target wealth on or before n trials, when betting using Kelly criterion

        Parameters
        ----------
        target: float
            The value of target wealth
        n_trails: int
            The number of trials on or before which target wealth is reached
        f: float
            The fraction of wealth that will be bet at each round

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

        x = np.log(target)
        y = -g / g_var
        #TODO: check if initial wealth has been properly accounted for.
        # Can output probabilities greater than 1.
        alpha = (-g * (np.sqrt(n_trials))) / np.sqrt(g_var)
        beta = (np.log(target)- self.logwealth) / (np.sqrt(g_var) * np.sqrt(n_trials))
        N = norm(0.0, 1.0)

        prob = (1 - N.cdf(alpha + beta)) + (np.exp(-2.0 * x * y)) * N.cdf(alpha - beta)
        return prob

    def prob_exceeding_target(self, target, n_trials, f=None):
        """
        Returns the probability of having wealth equal to or more than the target amount at the end of n_trials

        Parameters
        ----------
        target: float
            The value of target wealth
        n_trials: int
            The number of trials on or before which target wealth is reached
        f: float
            The fraction of wealth bet at each round
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


        #TODO: check if initial wealth has been properly accounted for
        alpha = (-g * (np.sqrt(n_trials))) / np.sqrt(g_var)
        beta = (np.log(target)-self.logwealth) / (np.sqrt(g_var) * np.sqrt(n_trials))
        N = norm(0.0, 1.0)

        prob = 1 - N.cdf(alpha + beta)
        return prob

    def prob_fractional_loss(self,target_fraction,f=None):
        """
        Returns the probability of being reduced to a given fraction of the initial wealth

        Parameters
        ----------
        target_fraction: float
            The fraction of wealth for which the probability of occurrence will be calculated for
        f: float
            The fraction of wealth bet at each round

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
        prob = target_fraction ** ((2 * g) / g_var)

        return prob

    def gamble(self, n_trails, f):
        """
        Runs a series of random trials where a fraction of wealth is bet at each trial and returns the log of the
        remain wealth

        Parameters
        ----------
        n_trails: int
            the number of random trials
        f: float
            the fraction of wealth bet at each trial

        Returns
        -------
        logwealth: float
            the log of the wealth after n_trails
        """

        if f is None:
            f = self.f_kelly

        outcomes = np.random.choice((-self.a, self.b),size=n_trails,p=(1-self.p, self.p))
        self.logwealth = self.logwealth + np.sum(np.log(1+outcomes*f))

        return self.logwealth

    def predict_gamble(self, n_trails, f):
        """
        Predict (estimate) the logarithm of wealth after a series of random trials where a fraction of wealth is bet
        at each trial.

        Parameters
        ----------
        n_trails: int
            the number of random trials
        f: float
            the fraction of wealth bet at each trial

        Returns
        -------
        logwealth: float
            the expected log of the wealth after n_trials
        """

        return self.logwealth + n_trails * self.expected_log_return(f)

def main():
    """
    Run various test to check these functions
    """
    coin = SingleCoinBetting(p=0.51,initial_logwealth=0)
    n_trails = 10000

    npt.assert_almost_equal(coin.prob_reaching_target(2, n_trails), 0.9214, decimal=4,
                            err_msg="Couldn't reproduce example from book.")
    npt.assert_almost_equal(coin.prob_exceeding_target(2, n_trails), 0.7433, decimal=4,
                            err_msg="Couldn't reproduce example from book.")
    npt.assert_almost_equal(coin.prob_fractional_loss(target_fraction=0.7), 0.7, decimal=1,
                            err_msg="Couldn't reproduce example from book.")

if __name__ == '__main__':
    import numpy.testing as npt
    main()
