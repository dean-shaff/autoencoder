from autoencoder import AutoEncoder 
import theano.tensor as T 
from SGDautoencoder import SGDAutoEncoder

if __name__ == '__main__':
	
	x = T.matrix('x')
	model = AutoEncoder()