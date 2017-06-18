import sys
sys.path.append("./")
from bayes_opt import BayesianOptimization

ranges = {	'rounds':[10, 3000], 'eta':[0.01,0.3], 'p' = [0.05,0.4]
			'alpha': [0.1, 4], 'beta': [0.2, 8], 'mult': [0.4,1.2]}

values0 = {	'ROUNDS':250, 'eta':0.05,'p' = 0.165
			'alpha': 0.25, 'beta':1.5, 'mult': 0.8}
logloss0 = 0.#Fill-in!

values1 = {	'ROUNDS':1500, 'eta':0.07,'p' = 0.25
			'alpha': 1.0, 'beta':1.0, 'mult': 1.0}
logloss1 = 0.#Fill-in!

# add more

funcvalues = {values0:logloss0, values1:logloss1}

def function(rounds, eta, p, alpha, beta, mult)
	values = { 	'rounds':rounds, 'eta':eta, 'p':p,
				'alpha':alpha, 'beta':beta, 'mult':mult}
	if values in funcvalues:
		return funcvalues[values];
	else
		print values
		return 0
	
bo = BayesianOptimization(function, ranges)
bo.initialize() # TODO.


# # Lets find the maximum of a simple quadratic function of two variables
# # We create the bayes_opt object and pass the function to be maximized
# # together with the parameters names and their bounds.
# bo = BayesianOptimization(lambda x, y: -x ** 2 - (y - 1) ** 2 + 1,
                          # {'x': (-4, 4), 'y': (-3, 3)})

# # One of the things we can do with this object is pass points
# # which we want the algorithm to probe. A dictionary with the
# # parameters names and a list of values to include in the search
# # must be given.
# bo.explore({'x': [-1, 3], 'y': [-2, 2]})

# # Additionally, if we have any prior knowledge of the behaviour of
# # the target function (even if not totally accurate) we can also
# # tell that to the optimizer.
# # Here we pass a dictionary with 'target' and parameter names as keys and a
# # list of corresponding values
# bo.initialize(
    # {
        # 'target': [-1, -1],
        # 'x': [1, 1],
        # 'y': [0, 2]
    # }
# )

# # Once we are satisfied with the initialization conditions
# # we let the algorithm do its magic by calling the maximize()
# # method.
# bo.maximize(init_points=5, n_iter=15, kappa=2)

# # The output values can be accessed with self.res
# print(bo.res['max'])

# # If we are not satisfied with the current results we can pickup from
# # where we left, maybe pass some more exploration points to the algorithm
# # change any parameters we may choose, and the let it run again.
# bo.explore({'x': [0.6], 'y': [-0.23]})

# # Making changes to the gaussian process can impact the algorithm
# # dramatically.
# gp_params = {'kernel': None,
             # 'alpha': 1e-5}

# # Run it again with different acquisition function
# bo.maximize(n_iter=5, acq='ei', **gp_params)

# # Finally, we take a look at the final results.
# print(bo.res['max'])
# print(bo.res['all'])