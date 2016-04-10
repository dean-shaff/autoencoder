import numpy as np 
import os 
import theano.tensor as T 
import theano.tensor.signal.pool as pool 
import theano 
from scipy.misc import imread 
import matplotlib.pyplot as plt 
import time 

def file_iter(dir_name):
	"""
	Assumes you've already moved to the folder in question.
	"""
	for file_name in os.listdir(dir_name):
		yield file_name

def load_data(reduce_dim,dir_name,up_to=None,max_pool=True):
	"""
	Load image data from a folder into a single theano shared variable. 
	The directory should only contain the images to be used. 
	args:
		-reduce_dim: a tuple or list containing the dimensionality of the crop to make.
			if max_pool is false, this gets used for cropping. 
			if max_pool is true, this gets used for max pooling 
		-dir_name: the directory from which to extract data 
	"""
	if max_pool:
		x = T.matrix('x')
		fpool = theano.function(inputs=[x],outputs= pool.pool_2d(x,reduce_dim,ignore_border=True))
		csv_file = "img{}_{}-{}maxpool.csv"
	else:
		csv_file = "img{}_{}-{}.csv"
		fpool = None 
	imgs = []
	os.chdir(dir_name)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	file_list = os.listdir(dir_name)
	file_list = [name for name in file_list if ".png" in name] #only want png files 
	print(file_list[:10])
	if (up_to > len(file_list)):
		print("Can't iterate through more files than actually exist!")
		return None 
	elif up_to == None:
		up_to = len(file_list)
	dim = ""
	t0 = time.time()
	for i, img_name in enumerate(file_list):
		t1 = time.time() 
		if i == up_to:
			break
		im = imread(img_name)
		r = im[:,:,0]
		b = im[:,:,1]
		g = im[:,:,2]
		grey = 0.2126*r + 0.7152*g + 0.0722*b
		if not max_pool:
			w = grey.shape[0] #this is inefficient
			h = grey.shape[1]
			w_c = reduce_dim[0] #width crop
			h_c = reduce_dim[1] #height crop
			dim = (w_c, h_c)
			crop = grey[int(w/2)-int(w_c/2):int(w/2)+int(w_c/2),int(h/2)-int(h_c/2):int(h/2)+int(h_c/2)]
		if max_pool:
			crop = fpool(grey)
			dim = crop.shape
			# ax.imshow(crop,cmap='gray')
			# plt.show()
		crop /= np.amax(crop)
		imgs.append(crop.flatten())

		print("Time for this iteration: {:.2f}, only {} more to go!".format(time.time()-t1, up_to-i))
	csv_file = csv_file.format(str(up_to), str(dim[0]), str(dim[1]))
	np.savetxt(csv_file, np.asarray(imgs), delimiter=",")
	print("Total time iterating through images and saving: {:.2f}".format(time.time()-t0))

def load_csv(csv_file, dir_name):
	os.chdir(dir_name)
	if (os.path.isfile(csv_file)):
		t0 = time.time()
		imgs = np.loadtxt(csv_file,delimiter=",")
		# print(imgs.shape)
		imgs_shared = theano.shared(imgs,borrow=True)
		print("Shared variable created. Took {:.2f} seconds".format(time.time() - t0))
		return imgs_shared
	else:
		print("The specified file doesn't exist")
		return None 

if __name__ == '__main__':
	dir_name1 = "/Users/dean/python_stuff_mac/machine_learning/autoencoder/data/jterm_vid"
	# csv_file = "imgSMALLmaxpool.csv"
	load_data((20,20),dir_name1,up_to=None,max_pool=True)
	# load_csv(csv_file,dir_name1)
