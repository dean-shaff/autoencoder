import theano 
import theano.tensor as T 
import time
import numpy as np 
from image_processor import load_csv 
import cPickle as pickle 
import matplotlib.pyplot as plt 

class HiddenLayer(object):

	def __init__(self,input,n_in,n_out,rng):
		self.n_in = n_in
		self.n_out = n_out
		self.W = theano.shared(
					value=rng.rand(n_in, n_out),
					name='W',
					borrow=True)

		self.b = theano.shared(
					value=rng.rand(n_out,),
					name='b',
					borrow=True)

		# def single_input(x):
		# self.output, updates = theano.scan(fn= lambda x: T.dot(self.W,x) + self.b,
		# 						 			sequences=[input]) #you have to iterate thrugh your input variables. 
		
		self.output = T.dot(input,self.W) + self.b
		self.params = [self.W, self.b]

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
		self.input = input 
		if len(dim) < 3:
			print("You're not building an autoencoder!")

		layers = [HiddenLayer(input,dim[0],dim[1],rng)]
		params = layers[0].params

		for i in xrange(len(dim)-2):
			layer = HiddenLayer(T.nnet.sigmoid(layers[-1].output),dim[i+1],dim[i+2],rng)
			layers.append(layer)
			params += [layer.W, layer.b]

		self.params = params 
		self.layers = layers 
		# print(self.layers[0].n_out, self.layers[1].n_out)
		self.final_output = self.layers[-1].output #should use sigmoid, not softmax. This isnt classification! 
		# self.pred = T.argmax(self.final_output, axis=1)

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

class SGDAutoEncoder(object):

	def __init__(self,model,dataset):
		"""
		Dataset should already be theano shared variables. 
		It should already be flattened (vectorized)
		"""
		self.model = model 
		self.cost = self.model.square_error()
		self.dataset = dataset 
		self.dataset_size = dataset.get_value().shape[0]
		self.fig = plt.figure()
		self.ax1 = self.fig.add_subplot(211)
		self.ax2 = self.fig.add_subplot(212)
		# plt.ion()

		# self.index = T.lscalar()

	def compile_functions(self,**kwargs): #minibatch_size,lr):
		"""
		compile training and testing functions.
		kwargs:
			lr: learning rate (0.001)
			mb_size: minibatch_size (10)
		"""
		lr = kwargs.get('lr',0.001)
		minibatch_size = kwargs.get('mb_size',10)
		print("Starting to compile theano functions. Cool ya jets...")
		t0 = time.time() 
		index = T.lscalar()
		gparams = [T.grad(self.cost,param) for param in self.model.params]
		updates = [(param, param-lr*gparam) for param, gparam in zip(self.model.params,gparams)]
		self.train_model = theano.function(
			inputs = [index],
			outputs = self.cost,
			updates = updates,
			givens = {
				x: self.dataset[index * minibatch_size: (index + 1) * minibatch_size]
			}
 		)
		# self.feed_thru = theano.function(
		# 	inputs = [index],
		# 	outputs = self.model.final_output,
		# 	givens = {
		# 		x: self.dataset[index * minibatch_size:(index + 1) * minibatch_size]
		# 	}
		# )
		self.feed_thru = theano.function(
			inputs = [index],
			outputs = self.model.final_output,
			givens = {
				x: self.dataset[index:index+1]
			}
		)
 		print("Compiling functions took {:.2f} seconds.".format(time.time() - t0))
 		return self.train_model, self.feed_thru

 	def train_model(self, **kwargs):
 		"""
		train the model 
		kwargs:
			lr: learning rate (0.001)
			mb_size: minibatch_size (10)
			n_epochs: number of epochs (5)
		"""
		lr = kwargs.get('lr',0.001)
		minibatch_size = kwargs.get('mb_size',10)
		n_epochs = kwargs.get('n_epochs',5)
		train_batches = self.dataset.get_value(borrow=True).shape[0] // minibatch_size
		trainer, feeder = self.compile_functions(**kwargs)
		# trainer = self.compile_functions(**kwargs)
		for epoch in xrange(n_epochs):
			for mb_index in xrange(train_batches):
				cost = trainer(mb_index)
			print(cost, epoch)
			if (epoch % 500 == 0):
				self.plot_result((109,192))

	def plot_result(self,og_dim):
		"""
		plot a result during training.
		args:
			-og_dim: a tuple or list corresponding to the original dimensions of the images. 

		"""
		try:
			feed_thru = self.feed_thru
		except AttributeError:
			print("You haven't called compile_functions yet!")

		# test = self.dataset.get_value()[np.random.random_integers(self.dataset_size-1)]
		# test = self.dataset.get_value()[0:2]
		# print(test)
		# print(test.shape)
		# self.fig = plt.figure()
		rando = np.random.random_integers(self.dataset_size-1)
		test = self.dataset.get_value()[rando]
		result = feed_thru(rando).reshape(og_dim)
		print(result)
		print(test.reshape(og_dim))
		# ax1 = self.fig.add_subplot(211)
		# ax2 = self.fig.add_subplot(212)
		# ax1.imshow(result)
		# ax2.imshow(test.reshape(og_dim))

		self.ax1.imshow(result)
		self.ax2.imshow(test.reshape(og_dim))
		plt.pause(0.001)
		raw_input(">>> ")
		# ax1.clear()
		# ax2.clear()

		self.ax1.clear()
		self.ax2.clear()

	def save_model(self,file_name):
		"""
		save the current model
		args:
			-file_name: name of pickle file to save in. 
		"""
		file_obj = open(file_name,'wb')
		pickle.dump(self.model,file_obj)
		file_obj.close()

if __name__ == '__main__':
	#dataset processing:
	dir_name1 = "/Users/dean/python_stuff_mac/machine_learning/autoencoder/data/jterm_vid"
	csv_file = "imgSMALLmaxpool.csv"
	# load_data((200,200),dir_name1,csv_file,None)
	shared = load_csv(csv_file,dir_name1)
	xtest = shared.get_value()[0]
	dim = xtest.shape[0]
	# print(dim)
	# print(shared.get_value().shape)
	x = T.matrix('x')
	rng = np.random.RandomState(1234)
	auto = AutoEncoder(x,[dim,500,dim],rng)
	sgd = SGDAutoEncoder(auto,shared)
	sgd.train_model(n_epochs=5000)
	# print(auto.params)
	# print(auto.layers[0].n_out,auto.layers[1].n_out)
















