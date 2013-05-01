
from inference import *
from control import *
from trace import *
from erp import *
from memoize import *

from datetime import datetime

samples = 150
lag = 20
runs = 5
errorTolerance = 0.07

def test(name, estimates, trueExpectation, tolerance=errorTolerance):

	print "test: {0} ...".format(name),

	errors = map(lambda estimate: abs(estimate - trueExpectation), estimates)
	meanAbsError = mean(errors)
	if meanAbsError > tolerance:
		print "failed! True mean: {0} | Test mean: {1}".format(trueExpectation, mean(estimates))
	else:
		print "passed."

def mhtest(name, computation, trueExpectation, tolerance=errorTolerance):
	#test(name, repeat(runs, lambda: expectation(computation, traceMH, samples, lag)), trueExpectation, tolerance)
	test(name, repeat(runs, lambda: expectation(computation, LARJMH, samples, 0, None, lag)), trueExpectation, tolerance)

def larjtest(name, computation, trueExpectation, tolerance=errorTolerance):
	test(name, repeat(runs, lambda: expectation(computation, LARJMH, samples, 10, None, lag)), trueExpectation, tolerance)

def eqtest(name, estvalues, truevalues, tolerance=errorTolerance):
	print "test: {0} ...".format(name),
	assert(len(estvalues) == len(truevalues))
	for i in xrange(len(estvalues)):
		estvalue = estvalues[i]
		truevalue = truevalues[i]
		if abs(estvalue - truevalue) > tolerance:
			print "failed! True value: {0} | Test value: {1}".format(truevalue, estvalue)
			return
	print "passed."

if __name__ == "__main__":

	d1 = datetime.now()

	print "starting tests..."


	"""
	ERP tests
	"""

	test("flip sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: flip(0.7)))), \
		  0.7)

	mhtest("flip query", \
			lambda: flip(0.7), \
			0.7)

	test("uniform sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: uniform(0.1, 0.4)))), \
		  0.5*(.1+.4))

	mhtest("uniform query", \
			lambda: uniform(.1, .4), \
			0.5*(.1+.4))

	test("multinomial sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: multinomialDraw([.2,.3,.4], [0.2, 0.6, 0.2])))), \
		  0.2*.2 + 0.6*.3 + 0.2*.4)

	mhtest("multinomial query", \
			lambda: multinomialDraw([.2,.3,.4], [0.2, 0.6, 0.2]), \
			0.2*.2 + 0.6*.3 + 0.2*.4)

	eqtest("multinomial lp", \
		[multinomial_logprob(0, [0.2, 0.6, 0.2]), \
		 multinomial_logprob(1, [0.2, 0.6, 0.2]), \
		 multinomial_logprob(2, [0.2, 0.6, 0.2])], \
		[math.log(0.2), math.log(0.6), math.log(0.2)])

	test("gaussian sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: gaussian(0.1, 0.5)))), \
		  0.1)

	mhtest("gaussian query", \
			lambda: gaussian(0.1, 0.5), \
			0.1)

	eqtest("gaussian lp", \
		[gaussian_logprob(0, 0.1, 0.5), \
		 gaussian_logprob(0.25, 0.1, 0.5), \
		 gaussian_logprob(0.6, 0.1, 0.5)], \
		[-0.2457913526447274, -0.27079135264472737, -0.7257913526447274])

	test("gamma sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: gamma(2, 2)/10))), \
		  0.4)

	mhtest("gamma query", \
			lambda: gamma(2, 2)/10, \
			0.4)

	eqtest("gamma lp", \
		[gamma_logprob(1, 2, 2), \
		 gamma_logprob(4, 2, 2), \
		 gamma_logprob(8, 2, 2)], \
		[-1.8862944092546166, -2.000000048134726, -3.306852867574781])

	test("beta sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: beta(2, 5)))), \
		  2.0/(2+5))

	mhtest("beta query", \
			lambda: beta(2, 5), \
			2.0/(2+5))

	eqtest("beta lp", \
		[beta_logprob(.1, 2, 5), \
		 beta_logprob(.2, 2, 5), \
		 beta_logprob(.6, 2, 5)], \
		[0.677170196389683, 0.899185234324094, -0.7747911992475776])

	test("binomial sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: binomial(.5, 40)/40.0))), \
		  0.5)

	mhtest("binomial query", \
			lambda: binomial(.5, 40)/40.0, \
			0.5)

	eqtest("binomial lp", \
		[binomial_logprob(15, .5, 40), \
		 binomial_logprob(20, .5, 40), \
		 binomial_logprob(30, .5, 40)], \
		[-3.3234338674089985, -2.0722579911387817, -7.2840211276953575])

	test("poisson sample", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: poisson(4)/10.0))), \
		  0.4)

	mhtest("poisson query", \
			lambda: poisson(4)/10.0, \
			0.4)

	eqtest("poisson lp", \
		[poisson_logprob(2, 4), \
		 poisson_logprob(5, 4), \
		 poisson_logprob(7, 4)], \
		[-1.9205584583201643, -1.8560199371825927, -2.821100833226181])


	"""
	Tests adapted from Church
	"""

	def flipSetTest():
		a = 1.0 / 1000
		condition(flip(a))
		return a
	mhtest("setting a flip", \
			flipSetTest, \
			1.0/1000, \
			tolerance=1e-15)


	def andConditionedOnOrTest():
		a = flip()
		b = flip()
		condition(a or b)
		return a and b
	mhtest("and conditioned on or", \
			andConditionedOnOrTest, \
			1.0/3)

	def biasedFlipTest():
		a = flip(0.3)
		b = flip(0.3)
		condition(a or b)
		return a and b
	mhtest("and conditioned on or, biased flip", \
			biasedFlipTest, \
			(0.3*0.3) / (0.3*0.3 + 0.7*0.3 + 0.3*0.7))


	def conditionedFlipTest():
		bitFlip = lambda fidelity, x: flip(fidelity if x else 1 - fidelity)
		hyp = flip(0.7)
		condition(bitFlip(0.8, hyp))
		return hyp
	mhtest("conditioned flip", \
			conditionedFlipTest, \
			(0.7*0.8) / (0.7*0.8 + 0.3*0.2))


	def randomIfBranchTest():
		if flip(0.7):
			return flip(0.2)
		else:
			return flip(0.8)
	mhtest("random 'if' with random branches, unconditioned", \
			randomIfBranchTest, \
			0.7*0.2 + 0.3*0.8)


	mhtest("flip with random weight, unconditioned", \
			lambda: flip(0.2 if flip(0.7) else 0.8), \
			0.7*0.2 + 0.3*0.8)


	def randomProcAppTest():
		proc = (lambda x: flip(0.2)) if flip(0.7) else (lambda x: flip(0.8))
		return proc(1)
	mhtest("random procedure application, unconditioned", \
			randomProcAppTest, \
			0.7*0.2 + 0.3*0.8)


	def conditionedMultinomialTest():
		hyp = multinomialDraw(['b', 'c', 'd'], [0.1, 0.6, 0.3])
		def observe(x):
			if flip(0.8):
				return x
			else:
				return 'b'
		condition(observe(hyp) == 'b')
		return hyp == 'b'
	mhtest("conditioned multinomial", \
			conditionedMultinomialTest, \
			0.357)


	def recursiveStochasticTailTest():
		def powerLaw(prob, x):
			if flip(prob, isStructural=True):
				return x
			else:
				return powerLaw(prob, x+1)
		a = powerLaw(0.3, 1)
		return a < 5
	mhtest("recursive stochastic fn, unconditioned (tail recursive)", \
			recursiveStochasticTailTest, \
			0.7599)

	def recursiveStochasticTest():
		def powerLaw(prob, x):
			if flip(prob, isStructural=True):
				return x
			else:
				return 0 + powerLaw(prob, x+1)
		a = powerLaw(0.3, 1)
		return a < 5
	mhtest("recursive stochastic fn, unconditioned", \
			recursiveStochasticTest, \
			0.7599)


	def memoizedFlipTest():
		proc = mem(lambda x: flip(0.8))
		return all([proc(1), proc(2), proc(1), proc(2)])
	mhtest("memoized flip, unconditioned", \
			memoizedFlipTest, \
			0.64)


	def memoizedFlipConditionedTest():
		proc = mem(lambda x: flip(0.2))
		condition(any([proc(1), proc(2), proc(2), proc(2)]))
		return proc(1)
	mhtest("memoized flip, conditioned", \
			memoizedFlipConditionedTest, \
			0.5555555555555555)


	def boundSymbolInMemoizerTest():
		a = flip(0.8)
		proc = mem(lambda x: a)
		return all([proc(1), proc(1)])
	mhtest("bound symbol used inside memoizer, unconditioned", \
			boundSymbolInMemoizerTest, \
			0.8)


	def memRandomArgTest():
		proc = mem(lambda x: flip(0.8))
		return all([proc(uniformDraw([1,2,3])), proc(uniformDraw([1,2,3]))])
	mhtest("memoized flip with random argument, unconditioned", \
			memRandomArgTest, \
			0.6933333333333334)


	def memRandomProc():
		proc = (lambda x: flip(0.2)) if flip(0.7) else (lambda x: flip(0.8))
		memproc = mem(proc)
		return all([memproc(1), memproc(2)])
	mhtest("memoized random procedure, unconditioned", \
			memRandomProc, \
			0.22)


	def mhOverRejectionTest():
		def bitFlip(fidelity, x):
			return flip(fidelity if x else (1-fidelity))
		def innerQuery():
			a = flip(0.7)
			condition(bitFlip(0.8, a))
			return a
		return rejectionSample(innerQuery)
	mhtest("mh-query over rejection query for conditioned flip", \
			mhOverRejectionTest, \
			0.903225806451613)


	def transDimensionalTest():
		a = beta(1, 5) if flip(0.9, isStructural=True) else 0.7
		b = flip(a)
		condition(b)
		return a
	mhtest("trans-dimensional", \
			transDimensionalTest, \
			0.417)


	def transDimensionalLARJTest():
			a = beta(1, 5) if flip(0.9, isStructural=True) else 0.7
			b = flip(a)
			condition(b)
			return a
	larjtest("trans-dimensional (LARJ)", \
			  transDimensionalLARJTest, \
			  0.417)


	def memFlipInIfTest():
		a = mem(flip) if flip() else mem(flip)
		b = a()
		return b
	mhtest("memoized flip in if branch (create/destroy memprocs), unconditioned", \
			memFlipInIfTest, \
			0.5)


	"""
	Tests for things specific to new implementation
	"""


	def nativeLoopTest():
		accum  = 0
		for i in xrange(4):
			accum += flip()
		return accum / 4.0
	mhtest("native for loop", \
			nativeLoopTest, \
			0.5)


	def directConditionTest():
		accum = [0]
		def block(i):
			if i < 5:
				accum[0] += flip(0.5, conditionedValue=True)
			else:
				accum[0] += flip(0.5)
		ntimes(10, block)
		return accum[0] / 10.0
	mhtest("directly conditioning variable values", \
			directConditionTest, \
			0.75)


	print "tests done!"

	d2 = datetime.now()
	print "time: {0}".format((d2 - d1).total_seconds())

