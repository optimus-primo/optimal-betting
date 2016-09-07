from SingleCoin import SingleCoinBetting

coin = SingleCoinBetting(p=0.51, initial_logwealth=100)
print coin.logwealth
coin.gamble(10000)
#for i in xrange(1000):
#	coin.gamble(1)
	#print i, coin.logwealth
print coin.logwealth
