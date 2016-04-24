import theano.tensor as T 
import theano 
import numpy as np 
import time 
import os 
import cPickle as pickle 
import matplotlib.pyplot as plt 

from autoencoder import AutoEncoder,HiddenLayer
from dataset import Dataset 
from image_processor import load_csv, load_data

class SGDAutoEncoder(object):

	def __init__(self,model,dataset):
		"""
		Dataset should be "Dataset" object
		"""
		self.model = model 
		self.cost = self.model.square_error()
		self.dataset = dataset 
		self.shared = dataset.shared_array
		self.dataset_size = dataset.size 
		self.fig1 = plt.figure()
		# plt.axis('off')
		self.fig2 = plt.figure() 
		# plt.axis('off')
		self.ax1 = self.fig1.add_subplot(111)
		self.ax2 = self.fig2.add_subplot(111)
		# plt.tick_params(
		# 	    axis='x',          # changes apply to the x-axis
		# 	    which='both',      # both major and minor ticks are affected
		# 	    bottom='off',      # ticks along the bottom edge are off
		# 	    top='off',         # ticks along the top edge are off
		# 	    labelbottom='off')
		# plt.tick_params(
		# 	    axis='y',          # changes apply to the x-axis
		# 	    which='both',      # both major and minor ticks are affected
		# 	    bottom='off',      # ticks along the bottom edge are off
		# 	    top='off',         # ticks along the top edge are off
		# 	    labelbottom='off')
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
				x: self.shared[index * minibatch_size: (index + 1) * minibatch_size]
			}
 		)
		# self.feed_thru = theano.function(
		# 	inputs = [index],
		# 	outputs = self.model.final_output,
		# 	givens = {
		# 		x: self.shared[index * minibatch_size:(index + 1) * minibatch_size]
		# 	}
		# )
		self.feed_thru = theano.function(
			inputs = [index],
			outputs = self.model.h2.output,
			givens = {
				x: self.shared[index: index+1]
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
			save_rate: how often to save the model (50)
			epoch_offset: how long to wait before saving (0)
			plot: Plot or not (True)
			save: save or not (True)
		"""
		lr = kwargs.get('lr',0.01)
		minibatch_size = kwargs.get('mb_size',10)
		n_epochs = kwargs.get('n_epochs',5)
		save_rate = kwargs.get('save_rate',50)
		epoch_offset = kwargs.get('epoch_offset',0)
		plot = kwargs.get('plot',True)
		save = kwargs.get('save',True)
		train_batches = self.dataset.size // minibatch_size
		trainer, feeder = self.compile_functions(**kwargs)
		# trainer = self.compile_functions(**kwargs)
		init_epoch = self.model.cur_epoch
		for epoch in xrange(n_epochs):
			for mb_index in xrange(train_batches):
				cost = trainer(mb_index)
			print("Current cost per image: {:.5f}, epoch {}".format(cost/minibatch_size, epoch))
			if (epoch % save_rate == 0 and epoch > epoch_offset):

				factor = float(self.dataset.size) / float(n_epochs - epoch_offset)
				if plot:
					self.plot_result((77,128),num_photo=int((epoch - epoch_offset) * factor))
				self.model.cur_epoch = epoch + init_epoch
				if save:
					self.model.save_params("model_DS{}_epoch{}.pkl".format(self.dataset.name,self.model.cur_epoch))

	def plot_result(self,og_dim,num_photo=None):
		"""
		plot a result during training.
		args:
			-og_dim: a tuple or list corresponding to the original dimensions of the images. 

		"""
		try:
			feed_thru = self.feed_thru
		except AttributeError:
			print("You haven't called compile_functions yet! Calling now ..")
			_, feed_thru = self.compile_functions()

		# test = self.dataset.get_value()[np.random.random_integers(self.dataset_size-1)]
		# test = self.dataset.get_value()[0:2]
		# print(test)
		# print(test.shape)
		# self.fig = plt.figure()
		if num_photo == None:
			num_photo = np.random.random_integers(self.dataset_size-1)
		else:
			pass 
		print(num_photo)
		test = self.dataset.shared_array.get_value()[num_photo]
		# print(test)
		result = feed_thru(num_photo).reshape(og_dim)
		# print(result)
		self.ax1.xaxis.set_visible(False)
		self.ax1.yaxis.set_visible(False)
		self.ax2.xaxis.set_visible(False)
		self.ax2.yaxis.set_visible(False)

		self.ax1.imshow(result,cmap='gray')
		self.ax2.imshow(test.reshape(og_dim),cmap='gray')
		# plt.pause(0.001)
		# raw_input(">>> ")

		self.fig1.savefig('result_{:03d}.png'.format(num_photo),bbox_inches='tight',pad_inches=0)
		self.fig2.savefig('input_{:03d}.png'.format(num_photo),bbox_inches='tight',pad_inches=0)
		self.ax1.clear()
		self.ax2.clear()

	# def save_model(self,file_name):
	# 	"""
	# 	save the current model
	# 	args:
	# 		-file_name: name of pickle file to save in. 
	# 	"""
	# 	print("Starting to save model")
	# 	t0 = time.time()
	# 	file_obj = open(file_name,'wb')
	# 	pickle.dump(self.model,file_obj)
	# 	file_obj.close()
	# 	print("Model saved. Took {} seconds.".format(time.time()-t0))

	# @staticmethod
	# def load_model(file_name):
	# 	"""
	# 	load an autoencoder model. 
	# 	"""
	# 	print("Loading model...")
	# 	t0 = time.time() 
	# 	file_obj = open(file_name,'r')
	# 	model = pickle.load(file_obj)
	# 	file_obj.close()
	# 	print("Model loaded. Took {:.2f} seconds".format(time.time() - t0 ))
	# 	return model 


if __name__ == '__main__':
	#dataset processing:
	dir_name1 = "/Users/dean/python_stuff_mac/machine_learning/autoencoder/data/v2vassignment2"
	csv_file = "img314_77-128maxpool.csv"
	csv_file_small = "img314_27-45maxpool.csv"
	# model_file = os.path.join(dir_name1,"model_DSimg314_77-128maxpool_epoch550.pkl")
	#========Dataset stuff ==========
	shared = load_csv(csv_file,dir_name1)
	dataset = Dataset(shared,csv_file)
	dim = dataset.vector_size
	#========Trainer building========
	x = T.matrix('x')
	rng = np.random.RandomState(1234)
	auto = AutoEncoder(x,[dim,500,dim],rng)
	# auto.save_params('test.pkl')
	# auto.load_set_params('model_DSimg314_27-45maxpool_epoch450.pkl')
	sgd = SGDAutoEncoder(auto,dataset)
	# sgd.model.cur_epoch = 450
	# for i in xrange(300):
	# 	sgd.plot_result((27,45),i)
	sgd.train_model(n_epochs=1500,save_rate=10,epoch_offset=0,minibatch_size=10,lr=0.001,plot=True,save=False)
	# print(auto.params)
	# print(auto.layers[0].n_out,auto.layers[1].n_out)



