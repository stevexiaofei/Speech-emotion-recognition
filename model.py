from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy as np

import theano
import theano.tensor as T
import lasagne
from theano import config
from theano import pp
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from new import *
from layer import *
from optimization import adadelta
from utils import net_params
SEED=12
np.random.seed(SEED)

def build_model_1():
	trng = RandomStreams(SEED)
	rng = np.random.RandomState(1234)
	use_noise = theano.shared(numpy_floatX(0.))
	x = T.tensor3('x', dtype=config.floatX)
	mask = T.matrix('mask',dtype=config.floatX)
	y = T.vector('y',dtype='int32')
	lr=T.scalar('lr')
	params=net_params()
################   build model ##########################
	L1=HiddenLayer(rng,x,64,32)
	#model_params=L1.params.copy()
	params.add(L1)
	L5=dropout_layer(L1.output,use_noise,trng)
	L2=lstm(rng,L5.output,mask,32,32)
	#model_params.update(L2.params)
	params.add(L2)
	L3=dropout_layer(L2.mean_pool_out,use_noise,trng)
	L4=LogisticRegression(L3.output,rng,32,4)
	params.add(L4)
	#model_params.update(L4.params)
########################################################
	cost=L4.negative_log_likelihood(y)
	errors=L4.errors(y)
	#grads=T.grad(cost,wrt=list(model_params.values()))
	grads=T.grad(cost,wrt=list(params.params.values()))
	f_grad_shared , f_update =adadelta(lr,params.params, grads, 
											x, mask, y, cost)
	f_error=theano.function([x,mask,y],outputs=errors,name='f_error')	
	return f_grad_shared,f_update,f_error,params,use_noise
def build_model_2():
	trng = RandomStreams(SEED)
	rng = np.random.RandomState(1234)
	use_noise = theano.shared(numpy_floatX(0.))
	x = T.tensor3('x', dtype=config.floatX)
	mask = T.matrix('mask',dtype=config.floatX)
	y = T.vector('y',dtype='int32')
	lr=T.scalar('lr')
	params=net_params()
################   build model ##########################
	L1=HiddenLayer(rng,x,64,32)
	#model_params=L1.params.copy()
	params.add(L1)
	L6=conv_layer(rng,L1.output,filters_shape=(32,1,11,32),input_shape=(None,1,None,32))
	mask_conv=L6.get_mask(mask)
	params.add(L6)
	L5=dropout_layer(L6.output,use_noise,trng)
	L2=lstm(rng,L5.output,mask_conv,32,32,pooling_type="mean_pooling")
	#model_params.update(L2.params)
	params.add(L2)
	L3=dropout_layer(L2.mean_pool_out,use_noise,trng)
	L4=LogisticRegression(L3.output,rng,32,4)
	params.add(L4)
	#model_params.update(L4.params)
########################################################
	cost=L4.negative_log_likelihood(y)
	errors=L4.errors(y)
	grads=T.grad(cost,wrt=list(params.params.values()))
	f_grad_shared , f_update =adadelta(lr,params.params, grads, 
											x, mask, y, cost)
	f_error=theano.function([x,mask,y],outputs=errors,name='f_error')	
	return f_grad_shared,f_update,f_error,params,use_noise
def build_model_3():
	trng = RandomStreams(SEED)
	rng = np.random.RandomState(1234)
	use_noise = theano.shared(numpy_floatX(0.))
	x = T.tensor3('x', dtype=config.floatX)
	mask = T.matrix('mask',dtype=config.floatX)
	y = T.vector('y',dtype='int32')
	lr=T.scalar('lr')
	params=net_params()
################   build model ##########################
	L1=HiddenLayer(rng,x,64,32)#wavenet_layer(x,rng,13,64,dilation_levels=4)# 
	params.add(L1)
	L6=conv_layer(rng,L1.output,filters_shape=(32,1,11,32),input_shape=(None,1,None,32))
	params.add(L6)
	mask_conv=L6.get_mask(mask)
	L8=conv_layer(rng,L1.output,filters_shape=(14,1,11,32),input_shape=(None,1,None,32))
	weight=L8.output.sum(axis=2)
	e_x=T.exp(weight-weight.max(axis=0,keepdims=True))
	e_x_mask=e_x*mask_conv
	attention=e_x_mask/e_x_mask.sum(axis=0,keepdims=True)

	L5=dropout_layer(L6.output,use_noise,trng)
	L2=lstm(rng,L5.output,mask_conv,32,32 )
	L7=lstm(rng,L5.output,mask_conv,32,32 ,go_backwards=True)
	#model_params.update(L2.params)
	params.add(L2)
	params.add(L7)
	LSTM_out=T.concatenate([L2.mean_pool_out,L7.mean_pool_out],axis=1)
	L3=dropout_layer(LSTM_out,use_noise,trng)
	L4=LogisticRegression(L3.output,rng,64,4)
	  
	params.add(L4)
	#model_params.update(L4.params)
########################################################
	cost=L4.negative_log_likelihood(y)
	updates_cost= lasagne.updates.adadelta(cost, list(params.params.values()))
	errors=L4.errors(y)
	#grads=T.grad(cost,wrt=list(model_params.values()))
	grads=T.grad(cost,wrt=list(params.params.values()))
	f_grad_shared , f_update =adadelta(lr,params.params, grads, 
											x, mask, y, cost)
	f_error=theano.function([x,mask,y],outputs=errors,name='f_error')	
	return f_grad_shared,f_update,f_error,params,use_noise
def build_model_4():
	trng = RandomStreams(SEED)
	rng = np.random.RandomState(1234)
	use_noise = theano.shared(numpy_floatX(0.))
	x = T.tensor3('x', dtype=config.floatX)
	mask = T.matrix('mask',dtype=config.floatX)
	y = T.vector('y',dtype='int32')
	lr=T.scalar('lr')
	params=net_params()
	x_one=np.ones(shape=(128,45,64)).astype(config.floatX)
	mask_one=np.ones(shape=(128,45)).astype(config.floatX)
################   build model ##########################
	L1=HiddenLayer(rng,x,64,32)#wavenet_layer(x,rng,13,64,dilation_levels=4)# 
	params.add(L1)
	L6=conv_layer(rng,L1.output,filters_shape=(32,1,11,32),input_shape=(None,1,None,32))
	params.add(L6)
	mask_conv=L6.get_mask(mask)
	
	L5=dropout_layer(L6.output,use_noise,trng)

	L2=lstm(rng,L5.output,mask_conv,32,32)
	params.add(L2)
	L8=conv_layer(rng,L2.h,filters_shape=(12,1,13,32),input_shape=(None,1,None,32))
	params.add(L8)
	weight=L8.output.max(axis=2)#++++++++++++++++++++++++++++++++++++++++++++++
#########################################################################
	e_x=T.exp(weight-weight.max(axis=0,keepdims=True))
	e_x_mask=e_x*mask_conv
	attention=e_x_mask/e_x_mask.sum(axis=0,keepdims=True)
	
	L7=lstm(rng,L5.output,mask_conv,32,32,go_backwards=True)
	params.add(L7)
	L9=conv_layer(rng,L2.h,filters_shape=(12,1,13,32),input_shape=(None,1,None,32))
	params.add(L9)
	weight=L9.output.max(axis=2)#++++++++++++++++++++++++++++++++++++++++++++
#########################################################################	
	e_x=T.exp(weight-weight.max(axis=0,keepdims=True))
	e_x_mask=e_x*mask_conv
	attention_bk=e_x_mask/e_x_mask.sum(axis=0,keepdims=True)
	
	LSTM_out=T.concatenate([(L2.h*attention[:,:,None]).sum(axis=0),(L7.h*attention_bk[:,:,None]).sum(axis=0)],axis=1)
	L3=dropout_layer(LSTM_out,use_noise,trng)
	L4=LogisticRegression(L3.output,rng,64,4)
	  
	params.add(L4)
	#model_params.update(L4.params)
########################################################
	cost=L4.negative_log_likelihood(y)
	errors=L4.errors(y)
	#grads=T.grad(cost,wrt=list(model_params.values()))
	grads=T.grad(cost,wrt=list(params.params.values()))
	f_grad_shared , f_update =adadelta(lr,params.params, grads, 
											x, mask, y, cost)
	f_error=theano.function([x,mask,y],outputs=errors,name='f_error')	
	return f_grad_shared,f_update,f_error,params,use_noise

def test_errors(f_error,datagene):
	error_list=[]
	for (x,mask),y,g in datagene.test_iterate():
		it_error=f_error(x,mask,y)
		#print(it_error.shape,y)
		error_list.append(it_error)
	return np.array(error_list).mean()
if __name__=='__main__':

	f_grad_shared,f_update,f_error,params,use_noise = build_model_1()
	datagene=datagenerate(portion2train=0.75)	
#####################   params setting ######################
	uidx =0
	max_epochs=40
	cost_list=[]
	error_list=[]
	patient=10
	lrate=0.0001
	bad_counter = 0
	estop=False
	load_params=False
	if load_params:
		params.load()
#####################################################
	for eidx in range(max_epochs):
		for (x,mask),y,g in datagene.train_iterate():
			use_noise.set_value(1.0)
			uidx+=1
			cost = f_grad_shared(x,mask,y)
			cost_list.append(cost)
			f_update(lrate)
			if np.mod(uidx,10)==0:
				print("%d update,cost:%.4f" % (uidx,cost))
			if np.mod(uidx,100)==0:
				use_noise.set_value(0.0)
				error=test_errors(f_error,datagene)
				error_list.append(error)
				print("test error:%.4f"%error)
				#print(error_list)
				if len(error_list)!=1 and error < np.array(error_list[:-1]).min():
					bad_counter=0
					print('save the parameter!')
					params.save()
				if len(error_list)>patient and error>= np.array(error_list)[-patient:].min():
					bad_counter+=1
					if bad_counter > patient:
						print("Early stop")
						estop=True
						break
		if	estop==True:
			break
	print("best error rate %.4f"%np.array(error_list).min())
			
	

		
		

		
