import trace

def prfor(iterable, block):
	"""
	'for' loop control structure suitable for use inside probabilistic programs.
	Invokes block for every element in iterable.
	"""
	for elem in iterable:
		trace.incrementLoopCounter(0)
		block(elem)

def prwhile(condition, block):
	"""
	'while' loop control structure suitable for use inside probabilistic programs.
	Invokes block while condition is true.
	"""
	cond = condition()
	while cond:
		trace.incrementLoopCounter(0)
		block()
		cond = condition()

def prmap(proc, iterable):
	"""
	Higher-order 'map' function suitable for use inside probabilistic programs.
	Transforms every element of iterable using proc, returning a new sequence object.
	"""
	def procwrapper(elem):
		trace.incrementLoopCounter(1)
		return proc(elem)
	return map(procwrapper, iterable)