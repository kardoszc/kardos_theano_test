from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
import time
# print time.time()
srng = RandomStreams(seed=int(time.time()))
start = time.time()
# srng = RandomStreams(seed=100)
rv_u = srng.uniform((2,2))
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
f_val0 = f()
f_val1 = f()  #different numbers from f_val0
print f_val0#.get_value()
print f_val1#.get_value()
g_val0 = g()  # different numbers from f_val0 and f_val1
g_val1 = g()  # same numbers as g_val0!
print g_val0#.get_value()
print g_val1#.get_value()
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
print time.time() - start
# rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
# rng_val.seed(89234)                         # seeds the generator
# rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng
# srng.seed(902340)  # seeds rv_u and rv_n with different seeds each 