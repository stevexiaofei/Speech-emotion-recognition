import theano
from theano import tensor as T
from theano import config
import lasagne
from lasagne.layers import DenseLayer,DimshuffleLayer,get_output, DropoutLayer, ReshapeLayer, InputLayer, MergeLayer,FlattenLayer,LSTMLayer, Conv1DLayer,ConcatLayer
from lasagne.objectives import categorical_crossentropy, aggregate,binary_crossentropy,binary_accuracy,categorical_accuracy

import numpy as np
from lasagne import utils
lasagne.random.set_rng(numpy.random.RandomState(123))
class attention_reduce_layer(MergeLayer):
	def	__init__(self,input,attention,mask,**kwargs):
		super(attention_reduce_layer,self).__init__([input,attention,mask],**kwargs)
		#self.mask=mask
	def get_output_shape_for(self,input_shapes):
		in1_shape,in2_shape,mask_shape=input_shapes
		return (in1_shape[0],in1_shape[2])
	def get_output_for(self,inputs,deterministic=False,**kwargs):
		input, attention,mask=inputs### input.shape=(batchsize,timestep,featdim) attention.shape=(batchsize,featuredim,timestep)
		weight=attention.sum(axis=1)
		weight=T.exp(weight-weight.max(axis=1,keepdims=True))
		weight=weight*mask
		weight=weight/weight.sum(axis=1,keepdims=True)
		return (input*weight[:,:,None]).sum(axis=1)
		
def model_build():
	x = T.tensor3('x', dtype=config.floatX)
	m = T.matrix('mask',dtype=config.floatX)
	y = T.vector('y',dtype='int32')
	g = T.vector('y',dtype='int32')
	output=InputLayer(shape=(None,None,64),input_var=x)
	mask=InputLayer(shape=(None,None),input_var=m)
	mask=DimshuffleLayer(mask,(1,0))
	output=DenseLayer(output,num_units=32,num_leading_axes=2)
	
	output=ReshapeLayer(output,([1],[2],[0]))
	output=Conv1DLayer(output,32,11,pad='same')
	output=ReshapeLayer(output,([0],[2],[1]))
	
	
	output_fw=LSTMLayer(output,32,peepholes=False,mask_input=mask)
	output_bw=LSTMLayer(output,32,peepholes=False,backwards=True,mask_input=mask)
	
	output_fw_attention=ReshapeLayer(output_fw,([0],[2],[1]))
	attention_fw=Conv1DLayer(output_fw_attention,12,11,pad='same')
	output_fw=attention_reduce_layer(output_fw,attention_fw,mask)
	
	output_bw_attention=ReshapeLayer(output_bw,([0],[2],[1]))
	attention_bw=Conv1DLayer(output_bw_attention,12,11,pad='same')
	output_bw=attention_reduce_layer(output_bw,attention_bw,mask)
	
	output=ConcatLayer([output_fw,output_bw],axis=1)
	out=DenseLayer(output,num_units=4,nonlinearity=T.nnet.softmax)
	mw=DenseLayer(output,num_units=1,nonlinearity=lasagne.nonlinearities.sigmoid)
	out_predictions,mw_predictions=get_output([out,mw])
	loss=categorical_crossentropy(out_predictions,y)
	loss = aggregate(loss, mode='mean')
	mw_loss=binary_crossentropy(mw_predictions,g)
	mw_loss=aggregate(mw_loss,mode='mean')
	total_loss=loss+mw_loss
	
	output_params=lasagne.layers.get_all_params(output, trainable=True) 
	out_params = output_params+out.get_params()
	mw_params = output_params+mw.get_params()
	params = output_params+mw.get_params()+ out.get_params()
	
	updates_mw= lasagne.updates.adadelta(mw_loss, mw_params)
	updates_class= lasagne.updates.adadelta(loss, out_params)								 
	updates = lasagne.updates.adadelta(total_loss, params)
										 
	out_predictions_deterministic,mw_predictions_deterministic=get_output([out,mw], deterministic=True)
	
	train_fn=theano.function([x,m,y,g],total_loss,updates=updates)
	train_mw_fn=theano.function([x,m,g],mw_loss,updates=updates_mw)
	train_class_fn=theano.function([x,m,y],loss,updates=updates_class)
	
	mw_acc=binary_accuracy(mw_predictions_deterministic,g)
	mw_acc=aggregate(mw_acc,mode='mean')
	
	class_acc=categorical_accuracy(out_predictions_deterministic,y)
	class_acc=aggregate(class_acc,mode='mean')
	
	test_class_fn=theano.function([x,m,y],class_acc)
	test_mw_fn=theano.function([x,m,g],mw_acc)
	
	return train_fn, train_mw_fn,train_class_fn, test_class_fn,test_mw_fn,params
	
	
train_fn, train_mw_fn,train_class_fn, test_class_fn,test_mw_fn,params=model_build()
from new import datagenerate
datagene=datagenerate(portion2train=0.8)	
#####################   params setting ######################
uidx =0
max_epochs=50
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
		uidx+=1
		cost= train_fn(x,mask,y,g)
		cost_list.append(cost)
		if np.mod(uidx,10)==0:
			print("%d update,cost:%.4f" % (uidx,cost))
		if np.mod(uidx,100)==0:
			sum=0;
			num=0
			for (x,m),y,g in datagene.test_iterate():
				sum+=test_class_fn(x,m,y)*len(y)
				num+=len(y)
			rr=sum/float(num)
			error_list.append(rr)
			print("recognition rate:%.4f"%rr)
			#print(error_list)
			if len(error_list)!=1 and rr > np.array(error_list[:-1]).max():
				bad_counter=0
				print('save the parameter!')
				np.savez('net_params',params)#params.save()
			if len(error_list)>patient and rr<= np.array(error_list)[-patient:].max():
				bad_counter+=1
				if bad_counter > patient:
					print("Early stop")
					estop=True
					break
	if	estop==True:
		break
print("best recognition  rate %.4f"%np.array(error_list).max())