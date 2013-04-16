
from inference import *
from control import *
from trace import *
from erp import *
from memoize import *

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
	test(name, repeat(runs, lambda: expectation(computation, traceMH, samples, lag)), trueExpectation, tolerance)
	# test(name, repeat(runs, lambda: expectation(computation, LARJMH, samples, 10, None, lag)), trueExpectation, tolerance)


if __name__ == "__main__":

	print "starting tests..."


	"""
	Tests adapted from Church
	"""


	test("random, no query", \
		  repeat(runs, lambda: mean(repeat(samples, lambda: flip(0.7)))), \
		  0.7)


	def flipSetTest():
		a = 1.0 / 1000
		condition(flip(a))
		return a
	mhtest("setting a flip", \
			flipSetTest, \
			1.0/1000, \
			tolerance=1e-15)


	mhtest("unconditioned flip", \
			lambda: flip(0.7), \
			0.7)


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
		if (flip(0.7)):
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


	def recursiveStochasticTest():
		def powerLaw(prob, x):
			if flip(prob, isStructural=True):
				return x
			else:
				return powerLaw(prob, x+1)
		a = powerLaw(0.3, 1)
		return a < 5
	mhtest("recursive stochastic fn, unconditioned", \
			recursiveStochasticTest, \
			0.7599)


	def memoizedFlipTest():
		proc = mem(lambda x: flip(0.8))
		return proc(1) and proc(2) and proc(1) and proc(2)
	mhtest("memoized flip, unconditioned", \
			memoizedFlipTest, \
			0.64)


	def memoizedFlipConditionedTest():
		proc = mem(lambda x: flip(0.2))
		condition(proc(1) or proc(2) or proc(2) or proc(2))
		return proc(1)
	mhtest("memoized flip, conditioned", \
			memoizedFlipConditionedTest, \
			0.5555555555555555)


	def boundSymbolInMemoizerTest():
		a = flip(0.8)
		proc = mem(lambda x: a)
		return proc(1) and proc(1)
	mhtest("bound symbol used inside memoizer, unconditioned", \
			boundSymbolInMemoizerTest, \
			0.8)


	def memRandomArgTest():
		proc = mem(lambda x: flip(0.8))
		return proc(uniformDraw([1,2,3])) and proc(uniformDraw([1,2,3]))
	mhtest("memoized flip with random argument, unconditioned", \
			memRandomArgTest, \
			0.6933333333333334)


	def memRandomProc():
		proc = (lambda x: flip(0.2)) if flip(0.7) else (lambda x: flip(0.8))
		memproc = mem(proc)
		return memproc(1) and memproc(2)
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


	def memFlipInIfTest():
		a = mem(flip) if flip() else mem(flip)
		b = a()
		return b
	mhtest("memoized flip in if branch (create/destroy memprocs), unconditioned", \
			memFlipInIfTest, \
			0.5)

# (check-test (repeat runs
#                     (lambda ()
#                       (mh-query samples lag
#                                 (define bb (make-dirichlet-discrete (list 0.5 0.5 0.5)))
#                                 (= (bb) (bb))
#                                 true )))
#             (lambda (b) (if b 1 0))
#             (/ (+ 1 0.5) (+ 1 (* 3 0.5)))
#             error-tolerance
#             "symmetric dirichlet-discrete, unconditioned." )

# (check-test (repeat runs
#                     (lambda ()
#                       (mh-query samples lag
#                                 (define bb (make-dirichlet-discrete (list 0.5 0.5)))
#                                 (= 0 (bb))
#                                 (= 0 (bb)) )))
#             (lambda (b) (if b 1 0))
#             (/ (+ 1 0.5) (+ 1 (* 2 0.5)))
#             error-tolerance
#             "symmetric dirichlet-discrete, conditioned." )

# (define crp-param 0.5)
# (check-test (repeat runs
#                     (lambda ()
#                       (mh-query samples lag
#                                 (define draw-type (make-CRP crp-param));(CRPmem 1.0 gensym))
#                                 (define class (mem (lambda (x) (draw-type))))
#                                 (eq? (class 'bob) (class 'mary))
#                                 (eq? (class 'bob) (class 'jim)))))
#             (lambda (x) (if x 1 0))
#             (/ 2.0 (+ 2.0 crp-param))
#             error-tolerance
#             "CRP third customer at first table, conditioned on second customer at first table." )

# (check-test (repeat runs
#                     (lambda ()
#                       (mh-query samples lag
#                                 (define draw-type (DPmem 1.0 gensym))
#                                 (define class (mem (lambda (x) (draw-type))))
#                                 (eq? (class 'bob) (class 'mary))
#                                 true)))
#             (lambda (x) (if x 1 0))
#             0.5
#             error-tolerance
#             "DPmem of gensym, unconditioned." )

# (define dirichlet-param 0.01)
# (define CRP-param 1.0)
# (check-test (repeat runs
#                     (lambda ()
#                       (mh-query samples lag
#                                 (define draw-type (make-CRP CRP-param))
#                                 (define obs (mem (lambda (type) (make-symmetric-dirichlet-discrete 3 dirichlet-param))))
#                                 (= (sample (obs (draw-type))) (sample (obs (draw-type))))
#                                 true)))
#             (lambda (x) (if x 1 0))
#             (+ (* (/ 1 (+ 1 CRP-param))  (/ (+ 1 dirichlet-param) (+ 1 (* 3 dirichlet-param))))   ;same crp table, same dirichlet draws
#                (* (/ CRP-param (+ 1 CRP-param))   (/ 1 3))) ;different crp tables, same dirichlet draws...
#             error-tolerance
#             "varying numbers of xrps inside mem." )


	"""
	Tests for things specific to Python version
	"""


	def nativeLoopTest():
		accum  = 0
		for i in xrange(10):
			accum += flip()
		return accum / 10.0
	mhtest("native for loop", \
			nativeLoopTest, \
			0.5)

	def ntimesTest():
		accum = [0]
		def block(i):
			accum[0] += flip()
		ntimes(10, block)
		return accum[0] / 10.0
	mhtest("ntimes control structure", \
			ntimesTest, \
			0.5)


	def foreachTest():
		accum = [0]
		def block(elem):
			accum[0] += flip()
		foreach(xrange(10), block)
		return accum[0] / 10.0
	mhtest("foreach control structure", \
			foreachTest, \
			0.5)


	def untilTest():
		accum = [0]
		i = [0]
		def block():
			i[0] += 1
			accum[0] += flip()
		until(lambda: i[0] == 10, block)
		return accum[0] / 10.0
	mhtest("until control structure", \
			untilTest, \
			0.5)


	def mapTest():
		return sum(map(lambda x: flip(), range(10))) / 10.0
	mhtest("map control structure", \
			mapTest, \
			0.5)


	def repeatTest():
		return sum(repeat(10, flip)) / 10.0
	mhtest("repeat control structure", \
			repeatTest, \
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

