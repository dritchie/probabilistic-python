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

def compareForwardToMH(computation, iters, postprocess = (lambda x : x)):
	forwardhist = distribForward(computation, iters, postprocess)
	print "Forward hist:"
	print forwardhist
	mhhist = distribMH(computation, iters, postprocess)
	print "MH hist:"
	print mhhist

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

###############################

if __name__ == "__main__":
	# compareForwardToMH(ones, 1000, len)
	# compareForwardToMH(sumOfTen, 1000)
	# compareForwardToMH(sumOfTenWhile, 1000)
	# compareForwardToMH(sumOfTenFor, 1000)
	compareForwardToMH(sumOfTenMap, 1000)
	#################
	# avglen = sum(map(lambda s: len(s), sample(constrainedOnes, 1000))) / 1000.0
	# print "Average length: {0}".format(avglen)
