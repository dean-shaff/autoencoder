import theano 
import theano.tensor as T 
import time
import numpy as np 
from image_processor import load_csv 
import cPickle as pickle 
import matplotlib.pyplot as plt 
from dataset import Dataset 
import time 

class HiddenLayer(object):

	def __init__(self,input,n_in,n_out,rng):

		self.input = input
		self.n_in = n_in
		self.n_out = n_out
		self.W = theano.shared(
					value=np.asarray(
                		rng.uniform(
		                    low=0, #-np.sqrt(6. / (n_in + n_out)),
		                    high=2.*np.sqrt(4. / (n_in + n_out)),
		                    size=(n_in, n_out)
                		),
                		dtype=theano.config.floatX
            		),
					name='W',
					borrow=True)

		self.b = theano.shared(
					value=np.zeros(n_out,).astype(theano.config.floatX),
					name='b',
					borrow=True)

		self.output = T.dot(input,self.W) + self.b #put input first so it iterates 
		self.sigoutput = T.nnet.sigmoid(self.output)
		self.params = [self.W, self.b]

	def set_params(self,params):
		# print(params)
		self.W.set_value(params[0]) #,name='W',borrow=True)
		self.b.set_value(params[1]) #,name='b',borrow=True)
		# self.params = [self.W, self.b]
		# self.output = T.dot(self.input, self.W) + self.b 

	# def __getstate__(self):
	# 	return [param.get_value() for param in self.params]

	# def __setstate__(self, state):
	# 	W, b = state
	# 	self.W = W 
	# 	self.b = b 

class AutoEncoder(object):

	def __init__(self,input,dim,rng):
		"""
		args:
			-input: a symbolic matrix for the input 
			-dim: list containing the dimensions of the layers of the network.
				eg [500,200,500] would be an autoencoder with a 200 node hidden layer. 
			-rng: Random number generator 

		TODO:
			should make an option for tied parameters. 
		"""
		self.cur_epoch = 0 
		self.input = input 
		self.dim = dim 
		if len(dim) < 3:
			print("You're not building an autoencoder!")

		self.h1 = HiddenLayer(input,dim[0],dim[1],rng)
		self.h2 = HiddenLayer(self.h1.sigoutput,dim[1],dim[2],rng)
		# self.params = self.layers[0].params

		# for i in xrange(len(dim)-2):
		# 	layer = HiddenLayer(self.layers[i].Toutput,dim[i+1],dim[i+2],rng)
		# 	self.layers.append(layer)
		# 	self.params += [layer.W, layer.b]


		self.layers = [self.h1, self.h2]
		self.params = self.h1.params + self.h2.params
		# self.params = params
		# print(self.params) 
		# self.layers = layers 
		# print(self.layers)
		# print(self.layers[0].n_out, self.layers[1].n_out)
		# self.final_output = self.layers[-1].output
		self.final_output = self.h2.output #should use sigmoid, not softmax. This isnt classification! 
		# self.pred = T.argmax(self.final_output, axis=1)

	def save_params(self,filename):
		print("Saving parameters...")
		t0 = time.time() 
		params = [param.get_value() for param in self.params]
		with open(filename, 'wb') as f:
			pickle.dump(params, f)
		print("Saving complete. Took {:.2f} seconds".format(time.time()-t0))

	def load_set_params(self, filename):
		print("Loading in parameters...")
		t0 = time.time()
		with open(filename, 'r') as f:
			params = pickle.load(f)
		self.set_params(params)
		print("Loading complete. Took {:.2f} seconds".format(time.time()-t0))

	def set_params(self,params):
		# self.params = []
		for i in xrange(len(self.layers)):
			self.layers[i].set_params(params[2*i:(2*i)+2])
			# self.params += self.layers[i].params

	def square_error(self):
		"""
		Return the square error for a minibatch 
		"""
		return T.mean((self.final_output - self.input)**2)

	def cross_entropy(self):
		"""
		return cross entropy error for a minibatch
		**not yet implemented**
		"""
		pass

	# def __getstate__(self):
	# 	params = [] 
	# 	for layer in self.layers:
	# 		params.append(layer.__getstate__())
	# 	return params 

	# def __setstate__(self,state):
	# 	self.params = [] 
	# 	self.layers = [] 
	# 	for i,params in enumerate(state):
	# 		self.layers.append
	# 		self.params += params 












