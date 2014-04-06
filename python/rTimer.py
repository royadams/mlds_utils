import time

class Timer():    
	def __enter__(self):
		self.start = time.clock()

	def __exit__(self, *args):
		self.end = time.clock()
		print(self.end - self.start)

