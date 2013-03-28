import inspect
import copy
import random

class _RVDatabaseRecord:
	"""
	Value item stored in the database
	"""

	def __init__(self, erp, params, val, logprob):
		self.erp = erp
		self.params = params
		self.val = val
		self.logprob = logprob
		self.active = True

	def __deepcopy__(self, memo):
		return _RVDatabaseRecord(self.erp, self.params[:], self.val, self.logprob)

class _RandomVariableDatabase:
	"""
	Database of randomness generated by a probabilistic program
	Modified/updated during inference over program traces
	"""

	def __init__(self):
		self._vars = {}
		self.logprob = 0
		self.newlogprob = 0		# From newly-added variables
		self.oldlogprob = 0		# From unreachable variables
		self.rootframe = None

	def __deepcopy__(self, memo):
		newdb = _RandomVariableDatabase()
		newdb.logprob = self.logprob
		newdb.oldlogprob = self.oldlogprob
		newdb.newlogprob = self.newlogprob
		newdb._vars = copy.deepcopy(self._vars, memo)
		return newdb

	def numVars(self):
		return len(self._vars)

	def chooseVariableRandomly(self):
		"""
		Returns a randomly-chosen variable from the trace
		Technically, returns a (name, record) pair
		"""
		name = random.choice(self._vars.keys())
		return (name, self._vars[name])

	def traceUpdate(self, computation):
		"""
		Run computation and update this database accordingly
		"""

		self.logprob = 0.0
		self.newlogprob = 0.0

		# First, mark all random values as 'inactive'; only
		# those reeached by the computation will become 'active'
		for record in self._vars.values():
			record.active = False

		# Mark that this is the 'root' of the current execution trace
		self.rootframe = inspect.currentframe()

		# Run the computation, which will create/lookup random variables
		retval = computation()

		# CLear out the root frame
		self.rootframe = None

		# Clean up any random values that are no longer reachable
		self.oldlogprob = 0.0
		for record in self._vars.values():
			if not record.active:
				self.oldlogprob += record.logprob
		self._vars = {name:record for name,record in self._vars.iteritems() if record.active}

		return retval

	def currentName(self, numFrameSkip):
		"""
		Return the current name, as determined by the execution
			trace of the current program.
		Skips the top 'numFrameSkip' stack frames that precede this
			function's stack frame (numFrameSkip+1 frames total)
		"""
		s = inspect.stack()[(numFrameSkip+1):]
		s.reverse()

		# Skip everything at the bottom up the stack until we get to
		# the root frame
		if self.rootframe != None:
			firsti = 0
			while s[firsti][0] != self.rootframe:
				firsti += 1
			s = s[(firsti+1):]

		name = ""
		for tup in s:
			f = tup[0]
			name += "{0}:{1}|".format(id(f.f_code), f.f_lineno)
		return name

	def lookup(self, name, erp, params):
		"""
		Looks up the value of a random variable.
		If this random variable does not exist, create it
		"""

		record = self._vars.get(name)
		if record == None or record.erp != erp:
			# Create new variable
			val = erp._sample_impl(params)
			ll = erp._logprob(val, params)
			self.newlogprob += ll
			record = _RVDatabaseRecord(erp, params, val, ll)
			self._vars[name] = record
		else:
			# Reuse existing variable
			if record.params != params:
				record.params = params
				record.logprob = erp._logprob(record.val, params)
		self.logprob += record.logprob
		record.active = True
		return record.val

	def getRecord(self, name):
		"""
		Simply retrieve the variable record associated with name
		"""
		return self._vars.get(name)

	def addFactor(self, num):
		"""
		Add a new factor into the log likelihood of the current trace
		"""

		self.logprob += num

"""
Global singleton instance
"""
_rvdb = None

def lookupVariableValue(erp, params, numFrameSkip):
	global _rvdb
	if _rvdb == None:
		return erp._sample_impl(params)
	else:
		name = _rvdb.currentName(numFrameSkip+1)
		return _rvdb.lookup(name, erp, params)

def newDatabase():
	global _rvdb
	_rvdb = _RandomVariableDatabase()

def getCurrentDatabase():
	return _rvdb

def setCurrentDatabase(newdb):
	global _rvdb
	_rvdb = newdb

def factor(num):
	global _rvdb
	if _rvdb != None:
		_rvdb.addFactor(num)