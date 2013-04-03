import cPickle


class MemoizedFunction:
	"""
	Wrapper around a function to memoize its results
	Source: http://stackoverflow.com/questions/4669391/python-anyone-have-a-memoizing-decorator-that-can-handle-unhashable-arguments
	This implementation allows us to memoize functions whose arguments can be arbitrary Python structures.
	However, it is slower for simple argument types such as numbers or strings.
	"""

	def __init__(self, func):
		self.func = func
		self.cache = {}

	def __call__(self, *args, **kwds):
		str = cPickle.dumps(args, 1)+cPickle.dumps(kwds, 1)
		if not self.cache.has_key(str):
			val =  self.func(*args, **kwds)
			self.cache[str] = val
			return val
		else:
			return self.cache[str]

def mem(func):
	return MemoizedFunction(func)
