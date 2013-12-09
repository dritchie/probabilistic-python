import random
import trace
import math
import copy

"""
A bunch of sampling/pdf code adapted from jschurch:
https://github.com/stuhlmueller/jschurch
"""

class RandomPrimitive:
	"""
	Abstract base class for all ERPs
	"""

	def _sample_impl(self, params):
		pass

	def _sample(self, params, isStructural, conditionedValue=None):
		# Assumes _sample is called from __call__ in
		# conrete subclasses
		return trace.lookupVariableValue(self, params, isStructural, 2, conditionedValue)

	def _logprob(self, val, params):
		pass

	def _proposal(self, currval, params):
		"""
		Subclasses can override to do more efficient proposals
		"""
		return self._sample_impl(params)

	def _logProposalProb(self, currval, propval, params):
		"""
		Subclasses can override to do more efficient proposals
		"""
		return self._logprob(propval, params)


class FlipRandomPrimitive(RandomPrimitive):
	"""
	ERP with Bernoulli distribution
	"""

	def __init__(self):
		pass

	def __call__(self, p=0.5, isStructural=False, conditionedValue=None):
		return self._sample([p], isStructural, conditionedValue)

	def _sample_impl(self, params):
		p = params[0]
		randval = random.random()
		return randval < p

	def _logprob(self, val, params):
		p = params[0]
		val = bool(val)
		prob = (p if val else 1.0-p)
		return math.log(prob)

	def _proposal(self, currval, params):
		return not(currval)

	def _logProposalProb(self, currval, propval, params):
		return 0.0 		# There's only one way to flip a binary variable


def gaussian_logprob(x, mu, sigma):
	return -.5*(1.8378770664093453 + 2*math.log(sigma) + (x - mu)*(x - mu)/(sigma*sigma))

def gaussian_logprob_sigmaSq(x, mu, sigmaSq):
	return -.5*(1.8378770664093453 + math.log(sigmaSq) + (x - mu)*(x - mu)/sigmaSq)

class GaussianRandomPrimitive(RandomPrimitive):
	"""
	ERP with Gaussian distribution
	"""

	def __init__(self):
		pass

	def __call__(self, mu, sigma, isStructural=False, conditionedValue=None):
		return self._sample([mu,sigma], isStructural, conditionedValue)

	def _sample_impl(self, params):
		return random.gauss(params[0], params[1])

	def _logprob(self, val, params):
		return gaussian_logprob(val, params[0], params[1])

	# Drift kernel
	def _proposal(self, currval, params):
		return random.gauss(currval, params[1])

	# Drift kernel
	def _logProposalProb(self, currval, propval, params):
		return gaussian_logprob(propval, currval, params[1])


gamma_cof = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]
def log_gamma(xx):
	global gamma_cof
	x = xx - 1.0
	tmp = x + 5.5
	tmp -= (x + 0.5)*math.log(tmp)
	ser = 1.000000000190015
	for j in xrange(5):
		x += 1
		ser += gamma_cof[j] / x
	return -tmp + math.log(2.5066282746310005*ser)

def gamma_logprob(x, a, b):
	return (a - 1)*math.log(x) - float(x)/b - log_gamma(a) - a*math.log(b);

class GammaRandomPrimitive(RandomPrimitive):
	"""
	ERP with Gamma distribution
	"""

	def __init__(self):
		pass

	def __call__(self, a, b, isStructural=False, conditionedValue=None):
		return self._sample([a,b], isStructural, conditionedValue)

	def _sample_impl(self, params):
		return random.gammavariate(params[0], params[1])

	def _logprob(self, val, params):
		return gamma_logprob(val, params[0], params[1])
	
	# TODO: Custom proposal kernel?
	
def log_beta(a, b):
	return log_gamma(a) + log_gamma(b) - log_gamma(a+b)

def beta_logprob(x, a, b):
	if x > 0 and x < 1:
		return (a-1)*math.log(x) + (b-1)*math.log(1-x) - log_beta(a,b)
	else:
		return -float('inf')

class BetaRandomPrimitive(RandomPrimitive):
	"""
	ERP with Beta distribution
	"""

	def __init__(self):
		pass

	def __call__(self, a, b, isStructural=False, conditionedValue=None):
		return self._sample([a,b], isStructural, conditionedValue)

	def _sample_impl(self, params):
		return random.betavariate(params[0], params[1])

	def _logprob(self, val, params):
		return beta_logprob(val, params[0], params[1])

	# TODO: Custom proposal kernel?

def binomial_sample(p, n):
	k = 0
	N = 10
	a = 0
	b = 0
	while n > N:
		a = 1 + n/2
		b = 1 + n-a
		x = random.betavariate(a, b)
		if x >= p:
			n = a-1
			p /= x
		else:
			k += a
			n = b-1
			p = (p-x) / (1.0-x)
	u = 0
	for i in xrange(n):
		u = random.random()
		if u < p:
			k += 1
	return int(k)

def g(x):
	if x == 0:
		return 1
	if x == 1:
		return 0
	d = 1 - x
	return (1 - (x * x) + (2 * x * math.log(x))) / (d * d)

def binomial_logprob(s, p, n):
	inv2 = 1.0/2
	inv3 = 1.0/3
	inv6 = 1.0/6
	if s >= n:
		return -float('inf')
	q = 1-p
	S = s + inv2
	T = n - s - inv2
	d1 = s + inv6 - (n + inv3) * p
	d2 = q/(s+inv2) - p/(T+inv2) + (q-inv2)/(n+1)
	d2 = d1 + 0.02*d2
	num = 1 + q * g(S/(n*p)) + p * g(T/(n*q))
	den = (n + inv6) * p * q
	z = num / den
	invsd = math.sqrt(z)
	z = d2 * invsd
	return gaussian_logprob(z, 0, 1) + math.log(invsd)

class BinomialRandomPrimitive(RandomPrimitive):
	"""
	ERP with binomial distribution
	"""

	def __init__(self):
		pass

	def __call__(self, p, n, isStructural=False, conditionedValue=None):
		return self._sample([p,n], isStructural, conditionedValue)

	def _sample_impl(self, params):
		return binomial_sample(params[0], params[1])

	def _logprob(self, val, params):
		return binomial_logprob(val, params[0], params[1])

	# TODO: Custom proposal kernel?

def poisson_sample(mu):
	k = 0
	while mu > 10:
		m = 7.0/8*mu
		x = random.gammavariate(m, 1)
		if x > mu:
			return int(k + binomial_sample(mu/x, int(m-1)))
		else:
			mu -= x
			k += m
	emu = math.exp(-mu)
	p = 1
	while p > emu:
		p *= random.random()
		k += 1
	return int(k-1)

def fact(x):
	t = 1
	while x > 1:
		t *= x
		x -= 1
	return t

def lnfact(x):
	if x < 1:
		x = 1
	if x < 12:
		return math.log(fact(round(x)))
	invx = 1.0 / x
	invx2 = invx*invx
	invx3 = invx2*invx
	invx5 = invx3*invx2
	invx7 = invx5*invx2
	ssum = ((x + 0.5) * math.log(x)) - x
	ssum += math.log(2*math.pi) / 2.0
	ssum += (invx / 12) - (invx / 360)
	ssum += (invx5 / 1260) - (invx7 / 1680)
	return ssum

def poisson_logprob(k, mu):
	return k * math.log(mu) - mu - lnfact(k)

class PoissonRandomPrimitive(RandomPrimitive):
	"""
	ERP with poisson distribution
	"""

	def __init__(self):
		pass

	def __call__(self, mu, isStructural=False, conditionedValue=None):
		return self._sample([mu], isStructural, conditionedValue)

	def _sample_impl(self, params):
		return poisson_sample(params[0])

	def _logprob(self, val, params):
		return poisson_logprob(val, params[0])

	# TODO: Custom proposal kernel?

def dirichlet_sample(alpha):
	ssum = 0
	theta = []
	for a in alpha:
		t = random.gammavariate(a, 1)
		theta.append(t)
		ssum += t
	for i in xrange(len(theta)):
		theta[i] /= ssum
	return theta

def dirichlet_logprob(theta, alpha):
	lopg = log_gamma(sum(alpha))
	for i in xrange(len(alpha)):
		logp += (alpha[i] - 1)*math.log(theta[i])
		logp -= log_gamma(alpha[i])
	return logp

class DirichletRandomPrimitive(RandomPrimitive):
	"""
	ERP with dirichlet distribution
	"""

	def __init__(self):
		pass

	def __call__(self, alpha, isStructural=False, conditionedValue=None):
		return self._sample(alpha, isStructural, conditionedValue)

	def _sample_impl(self, params):
		return dirichlet_sample(params)

	def _logprob(self, val, params):
		return dirichlet_logprob(val, params)

	# TODO: Custom proposal kernel?


def multinomial_sample(theta):
	result = 0
	x = random.random() * sum(theta)
	probAccum = 1e-6
	k = len(theta)
	while result < k and x > probAccum:
		probAccum += theta[result]
		result += 1
	return result - 1

def multinomial_logprob(n, theta):
	if n < 0 or n >= len(theta):
		return -float('inf')
	n = int(round(n))
	return math.log(theta[n]/sum(theta))

class MultinomialRandomPrimitive(RandomPrimitive):
	"""
	ERP with multinomial distribution
	"""

	def __init__(self):
		pass

	def __call__(self, theta, isStructural=False, conditionedValue=None):
		return self._sample(theta, isStructural, conditionedValue)

	def _sample_impl(self, params):
		return multinomial_sample(params)

	def _logprob(self, val, params):
		return multinomial_logprob(val, params)

	# Multinomial with currval projected out
	def _proposal(self, currval, params):
		newparams = copy.copy(params)
		newparams[currval] = 0.0
		return multinomial_sample(newparams)

	# Multinomial with currval projected out
	def _logProposalProb(self, currval, propval, params):
		newparams = copy.copy(params)
		newparams[currval] = 0.0
		return multinomial_logprob(propval, newparams)


class UniformRandomPrimitive(RandomPrimitive):
	"""
	ERP with uniform distribution
	"""

	def __init__(self):
		pass

	def __call__(self, lo, hi, isStructural=False, conditionedValue=None):
		return self._sample([lo, hi], isStructural, conditionedValue)

	def _sample_impl(self, params):
		return random.uniform(params[0], params[1])

	def _logprob(self, val, params):
		if val < params[0] or val > params[1]:
			return -float('inf')
		else:
			return -math.log(params[1] - params[0])

	# TODO: Custom proposal kernel?



"""
Singleton instances of all the ERP gerneators
"""

flip = FlipRandomPrimitive()
gaussian = GaussianRandomPrimitive()
gamma = GammaRandomPrimitive()
beta = BetaRandomPrimitive()
binomial = BinomialRandomPrimitive()
poisson = PoissonRandomPrimitive()
dirichlet = DirichletRandomPrimitive()
multinomial = MultinomialRandomPrimitive()
uniform = UniformRandomPrimitive()


"""
Random utilies built on top of ERPs
"""

def multinomialDraw(items, probs, isStructural=False):
	return items[multinomial(probs, isStructural=isStructural)]

def uniformDraw(items, isStructural=False):
	n = len(items)
	return items[multinomial(map(lambda x: 1.0/n, range(n)), isStructural=isStructural)]