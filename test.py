from probabilistic import *
import math
from collections import Counter

###############################

def distribForward(computation, iters, postprocess = (lambda x : x)):
	hist = Counter()
	i = 0
	while i < iters:
		i += 1
		hist[postprocess(computation())] += 1.0/iters
	return hist

def distribMH(computation, iters, postprocess = (lambda x : x)):
	hist = Counter()
	samps = sample(computation, iters)
	for s in samps:
		hist[postprocess(s)] += 1.0/iters
	return hist

def compareForwardToMHDists(computation, iters, postprocess = (lambda x : x)):
	forwardhist = distribForward(computation, iters, postprocess)
	print "Forward hist:"
	print forwardhist
	mhhist = distribMH(computation, iters, postprocess)
	print "MH hist:"
	print mhhist

def meanForward(computation, iters, postprocess = (lambda x : x)):
	mean = 0.0
	i = 0
	while i < iters:
		i += 1
		mean += postprocess(computation())
	return mean / iters

def meanMH(computation, iters, postprocess = (lambda x : x)):
	mean = 0.0
	samps = sample(computation, iters)
	for s in samps:
		mean += postprocess(s)
	return mean / iters

def compareForwardToMHMeans(computation, iters, postprocess = (lambda x : x)):
	forwardmean = meanForward(computation, iters, postprocess)
	print "Forward mean: {0}".format(forwardmean)
	mhmean = meanMH(computation, iters, postprocess)
	print "MH mean: {0}".format(mhmean)

###############################

def ones():
	if flip(0.75):
		return [1] + ones()
	else:
		return []

def constrainedOnes():
	seq = ones()
	factor(-math.pow(abs(len(seq) - 4), 10))
	return seq

def sumOfTen():
	num = 0
	# num += flip(0.5, conditionedValue=1)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	num += flip(0.5)
	return num

def constrainedSumOfTen():
	num = sumOfTen()
	condition(num >= 5)
	return num

def sumOfTenWhile():
	num = [0]
	i = [0]
	def block():
		i[0] += 1
		num[0] += flip(0.5)
	prwhile(lambda: i[0] < 10, block)
	return num[0]

def sumOfTenFor():
	num = [0]
	def block(i):
		num[0] += flip(0.5)
	prfor(xrange(10), block)
	return num[0]

def sumOfTenMap():
	return sum(prmap(lambda x: flip(0.5), range(10)))

def oneGaussian():
	return gaussian(10, 0.5)

def oneGamma():
	return gamma(9, 0.5)

def oneBeta():
	return beta(2, 2)

def oneBinomial():
	return binomial(0.5, 40)

def onePoisson():
	return poisson(10)

def memTest():
	func = mem(lambda x: flip(x))
	result1 = func(0.5)
	result2 = func(0.5)
	return result1 == result2

def noisyOr(a, astrength, b, bstrength, baserate):
	return (flip(astrength) and a) or \
		   (flip(bstrength) and b) or \
		   flip(baserate)

def sprinklerTest():

	weights = {"rain-str": 0.9, \
			   "rain-prior": 0.3, \
			   "sprinkler-str": 0.9, \
			   "sprinkler-prior": 0.2, \
			   "grass-baserate": 0.1}

	@mem
	def rain(day):
		return flip(weights["rain-prior"])

	@mem
	def sprinkler(day):
		return flip(weights["sprinkler-prior"])

	@mem
	def grassIsWet(day):
		return noisyOr(rain(day), weights["rain-str"], \
					   sprinkler(day), weights["sprinkler-str"], \
					   weights["grass-baserate"])

	condition(grassIsWet("day2"))

	return rain("day2")

###############################

if __name__ == "__main__":

	# compareForwardToMHDists(ones, 1000, len)
	# compareForwardToMHDists(sumOfTen, 1000)
	# compareForwardToMHDists(sumOfTenWhile, 1000)
	# compareForwardToMHDists(sumOfTenFor, 1000)
	# compareForwardToMHDists(sumOfTenMap, 1000)
	# compareForwardToMHMeans(oneGaussian, 10000)
	# compareForwardToMHMeans(oneGamma, 10000)
	# compareForwardToMHMeans(oneBeta, 10000)
	# compareForwardToMHMeans(oneBinomial, 10000)
	# compareForwardToMHMeans(onePoisson, 10000)

	# print "Average length: {0}".format(meanMH(constrainedOnes, 1000, lambda x: len(x)))

	# print "Average num 1s: {0}".format(meanMH(constrainedSumOfTen, 1000))

	# print memTest()

	print distribMH(sprinklerTest, 10000)

