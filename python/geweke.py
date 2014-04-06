import numpy as np
from scipy.stats import norm
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import utils

# Generates two sets of samples from the joint p(params,data) and 
# evaluates the statistics in testFuns for each of these sets
# Params:
# 	test_funs 					= List of functions over which expected values will be calculated
#	sample_prior 				= function which takes no arguments and returns a vector sample from the prior on you unobservable variables
#	sample_data 				= function which takes a vector of parameter values and returns a sample from the likelihood
#	sample_transition_kernel	= function which takes a vector of data values and a vector that is the previous sample from sample_transition_kernel and returns a vector that is a sample of the unobserved variables

np.random.seed(1142534)
# np.random.seed(1234)

def geweke(n_prior_samples,n_chain_samples,chain_skip,sample_prior,sample_data,sample_transition_kernel,test_max=True,test_vars=False,test_funs=[]):
	M = [[],[]]
	X= np.arange(0,1,1.0/n_chain_samples) + (.5/n_chain_samples)
	
	for s in range(n_prior_samples):
		M[0].append({})
		M[0][s]["unobservable"] = sample_prior()
		M[0][s]["observable"] = sample_data(M[0][s]["unobservable"])

	
	prev_unobservable =  sample_prior()
	ind = 0
	for s in range(n_chain_samples*chain_skip):
		prev_obs = sample_data(prev_unobservable)
		prev_unobservable = sample_transition_kernel(prev_obs,prev_unobservable)
		if s%chain_skip == 0:
			M[1].append({})
			M[1][ind]["observable"] = prev_obs
			M[1][ind]["unobservable"] = prev_unobservable
			ind+=1

	n_u_vars = M[0][0]["unobservable"].shape[0]
	n_o_vars = M[0][0]["observable"].shape[0]
	n_vars = n_u_vars + n_o_vars
	
	cur_pos = 0
	g0 = np.zeros(n_prior_samples)
	g1 = np.zeros(n_chain_samples)
	if test_max:
		for s in range(n_prior_samples):
			g0[s] = max(np.max(np.abs(M[0][s]["observable"])),np.max(np.abs(M[0][s]["unobservable"])))
		for s in range(n_chain_samples):
			g1[s] = max(np.max(np.abs(M[1][s]["observable"])),np.max(np.abs(M[1][s]["unobservable"])))
		Y = np.array([((g0 < x).mean() + .5*(g0 == x).mean()) for x in np.sort(g1)])
		plt.plot([0,1],[0,1],'--')
		plt.plot(X,Y)
		plt.savefig("pp_maxes.png")
		plt.clf()
				
	if test_vars:
		for v in range(n_vars):
			for s in range(n_prior_samples):
				g0[s] = np.hstack((M[0][s]["observable"],M[0][s]["unobservable"]))[v]
			for s in range(n_chain_samples):
				g1[s] = np.hstack((M[1][s]["observable"],M[1][s]["unobservable"]))[v]
			Y = np.array([(g0 < x).mean()+ .5*(g0 == x).mean() for x in np.sort(g1)])
			plt.plot([0,1],[0,1],'--')
			plt.plot(X,Y)
			plt.savefig("pp_ind_vars_%d.png"%v)
			plt.clf()
			
	for fun,f in zip(test_funs,range(len(test_funs))):
		g0 = [fun(np.hstack((S["observable"],S["unobservable"]))) for S in M[0]]
		g1 = [fun(np.hstack((S["observable"],S["unobservable"]))) for S in M[1]]
		Y = np.array([(g0 < x).mean()+ .5*(g0 == x).mean() for x in np.sort(g1)])
		plt.plot([0,1],[0,1],'--')
		plt.plot(X,Y)
		plt.savefig("pp_test_fun_%d.png"%f)
		plt.clf()
		
	# if test_products:
	# 	for s in range(n_prior_samples):
	# 		samp = np.hstack((M[0][s]["observable"],M[0][s]["unobservable"])).reshape((n_vars,1))
	# 		g0[cur_pos:(cur_pos+n_vars**2),s] = np.dot(samp,samp.T).reshape((n_vars**2,))
	# 	for s in range(n_chain_samples):
	# 		samp = np.hstack((M[1][s]["observable"],M[1][s]["unobservable"])).reshape((n_vars,1))
	# 		g1[cur_pos:(cur_pos+n_vars**2),s] = np.dot(samp,samp.T).reshape((n_vars**2,))
	# 	cur_pos += n_vars**2
		
	# g0_means = g0.mean(1)
	# g0_vars = g0.var(1)
	# # g_vars = (g**2).mean(2) - g_means**2
	# g1_means = g1.mean(1)
	# g1_vars = g1.var(1)
	# 
	# 
	# t_stats = (g0_means - g1_means)/np.sqrt(g0_vars/n_prior_samples + g1_vars/n_chain_samples)
	# rv = norm()
	# p_vals = 2*rv.cdf(-np.abs(t_stats))
	# 
	# for p in [.05,.01,.005,.001]:
	# 	print "%g percent of tests fail at p=%g"%(100*(p_vals<=p).mean(),p)
	# 	
	# ind = np.argmin(p_vals)
	# plt.hist(g0[ind,:])
	# plt.savefig("output/m0_dist.png")
	# plt.clf()
	# 
	# 
	# plt.hist(g1[ind,:])
	# plt.savefig("output/m1_dist.png")
	# plt.clf()
	
	# plt.plot(g0[10,100:200])
	# plt.savefig("output/m0_p_chain.png")
	# plt.clf()
	# 
	# plt.plot(g1[10,100:200])
	# plt.savefig("output/m1_p_chain.png")
	# plt.clf()
	
	# return p_vals,M