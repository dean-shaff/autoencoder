import os 

class Dataset(object):

	def __init__(self,shared_array,file_name,name=None):

		self.shared_array = shared_array
		self.file_name = file_name
		if name == None:
			self.name = os.path.splitext(self.file_name)[0]
		else:
			self.name = name 
		self.size = shared_array.get_value().shape[0]
		self.vector_size = shared_array.get_value()[0].shape[0]
