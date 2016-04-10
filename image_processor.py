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

def load_data(crop_dim,dir_name,csv_file,up_to,max_pool=False):
	"""
	Load image data from a folder into a single theano shared variable. 
	The directory should only contain the images to be used. 
	args:
		-crop_dim: a tuple or list containing the dimensionality of the crop to make.
		-dir_name: the directory from which to extract data 
	"""
	if max_pool:
		x = T.matrix('x')
		fpool = theano.function(inputs=[x],outputs= pool.pool_2d(x,(10,10),ignore_border=True))
	else:
		fpool = None 
	imgs = []
	os.chdir(dir_name)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	file_list = os.listdir(dir_name)
	if (up_to > len(file_list)):
		print("Can't iterate through more files than actually exist!")
		return None 
	elif up_to == None:
		up_to = len(file_list)
	t0 = time.time()
	for i, img_name in enumerate(file_list):
		t1 = time.time() 
		if i == up_to:
			break
		if ".csv" in img_name:
			continue
		im = imread(img_name)
		r = im[:,:,0]
		b = im[:,:,1]
		g = im[:,:,2]
		grey = 0.2126*r + 0.7152*g + 0.0722*b
		if not max_pool:
			w = grey.shape[0] #this is inefficient
			h = grey.shape[1]
			# print(w,h)
			w_c = crop_dim[0] #width crop 
			h_c = crop_dim[1] #height crop
			crop = grey[int(w/2)-int(w_c/2):int(w/2)+int(w_c/2),int(h/2)-int(h_c/2):int(h/2)+int(h_c/2)]
		if max_pool:
			crop = fpool(grey)
			# ax.imshow(crop,cmap='gray')
			# plt.show()
		crop /= np.amax(crop)
		imgs.append(crop.flatten())
		print("Time for this iteration: {:.2f}".format(time.time()-t1))

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
	csv_file = "imgSMALLmaxpool.csv"
	load_data((50,50),dir_name1,csv_file,100,max_pool=True)
	load_csv(csv_file,dir_name1)
