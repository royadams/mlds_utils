import numpy as np

def wishart(Sigma,n):
	d = Sigma.shape[0]
	assert n > d, "n=%d is less than d=%d"%(n,d)
	A = np.linalg.cholesky(Sigma)
	T = np.zeros((d,d))
	for i in range(d):
		T[i,0:(i)] = np.random.randn(i)
		T[i,i] = np.sqrt(np.random.chisquare(n-i))
	AT = np.dot(A,T)
	return np.dot(AT,AT.T)