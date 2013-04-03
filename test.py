from probabilistic import *
import math
from collections import Counter

###############################

def distribForward(computation, iters, postprocess = (lambda x : x)):
	hist = Counter()
	i = 0
	while i < iters:
		i += 1
		hist[postprocess(computation())] += 1
	return hist

def distribMH(computation, iters, postprocess = (lambda x : x)):
	hist = Counter()
	samps = sample(computation, iters)
	for s in samps:
		hist[postprocess(s)] += 1
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

