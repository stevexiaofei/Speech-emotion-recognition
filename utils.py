import numpy as np
from collections import OrderedDict
class net_params(object):
	def __init__(self):
		self.params={}
	def add(self,Layer):
		self.params.update(Layer.params)
	def load(self,path="net_params.npz"):
		params_dict=np.load(path)
		for it,ic in params_dict.items():
			self.params[it].set_value(ic)
	def save(self,path="net_params"):
		params_dict=OrderedDict()
		for it,ic in self.params.items():
			params_dict[it]=ic.get_value()
		np.savez(path,**params_dict)