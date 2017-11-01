import numpy as np

import theano
import theano.tensor as T
from theano import config
from theano.tensor.signal import conv
from theano.tensor.nnet import conv2d
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
########################################################################################################################################
	#										Hidden layer
#####################################################################################################################
def numpy_floatX(data):
	return np.asarray(data,dtype=config.floatX)
# start-snippet-1
def pp(pa,pb):
	return pa+"_"+pb
def ortho_weight(ndim):
	W = np.random.randn(ndim,ndim)
	u,s,v =np.linalg.svd(W)
	return u.astype(config.floatX)
class HiddenLayer(object):
	name="hiddenlayer"
	count=0
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
		HiddenLayer.count=HiddenLayer.count+1
		self.name=HiddenLayer.name+"_"+str(HiddenLayer.count)
		self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
		if W is None:
			W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
		if activation == theano.tensor.nnet.sigmoid:
			W_values *= 4

		W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
		b = theano.shared(value=b_values, name='b', borrow=True)
		
		lin_output = T.dot(input,W) +b
		self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
		self.params = {self.name+'_W':W,self.name+"_b":b}
	
########################################################################################################################################
	#										LogisticRegression layer
########################################################################################################################################
class LogisticRegression(object):
	"""Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""
	name="LogisticRegression"
	count=0
	def __init__(self,input,rng, n_in, n_out):
		""" Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

		"""
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		LogisticRegression.count=LogisticRegression.count+1
		self.name=LogisticRegression.name+"_"+str(LogisticRegression.count)
		#rng = np.random.RandomState(1234)
		W= np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
		W = theano.shared(value=W, name='W',borrow=True)
        # initialize the biases b as a vector of n_out 0s
		b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
		self.p_y_given_x = T.nnet.softmax(T.dot(input, W) + b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
		self.params = {self.name+'_W':W,self.name+"_b":b}
        # keep track of model input
		self.input = input

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
		"""
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
		"""

        # check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
		if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()
##################################################################################################
#										LSTM layer
###################################################################################################
class lstm(object):
	name="lstm"
	count=0
	def __init__(self,rng,input,mask,feat_dim,hidden_dim,weight=None,go_backwards=False):
		lstm.count=lstm.count+1
		self.name=lstm.name+"_"+str(lstm.count)
		self.input=input
		self.feat_dim=feat_dim
		self.mask=mask
		n_steps=input.shape[0]
		
		if input.ndim == 3:
			n_samples = input.shape[1]
		else:
			n_samples = 1
		def init_weight(rng,n_in,n_out):
			return np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
		W=np.concatenate([ init_weight(rng,feat_dim,hidden_dim) for i in range(4)],axis=1)
		W=theano.shared(value=W,name='W',borrow=True)
		U=np.concatenate([ ortho_weight(hidden_dim) for i in range(4) ],axis=1)
		U=theano.shared(value=U,name='W',borrow=True)
		b=np.zeros((4*hidden_dim,),dtype=theano.config.floatX)
		b=theano.shared(value=b,name='b',borrow=True)
		u= np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (feat_dim+hidden_dim)),
                    high=np.sqrt(6. / (feat_dim+hidden_dim)),
                    size=(feat_dim,)
                ),
                dtype=theano.config.floatX
            )
		u = theano.shared(value=u, name='u',borrow=True)
		def _slice(_x, n, dim):
			if _x.ndim == 3:
				return _x[:,:,n*dim:(n+1)*dim]
			else:
				return _x[:,n*dim:(n+1)*dim]
		def _step(m_,x_,h_,c_):
			preact = T.dot(h_,U)
			preact += x_

			i = T.nnet.sigmoid(_slice(preact, 0, hidden_dim))

			f = T.nnet.sigmoid(_slice(preact, 1, hidden_dim))
			o = T.nnet.sigmoid(_slice(preact, 2, hidden_dim))
			c = T.tanh(_slice(preact, 3, hidden_dim))
			c = f*c_ + i*c
			c = m_[:,None]*c + (1-m_)[:,None]*c_

			h = o*T.tanh(c)
			h = m_[:,None]*h + (1-m_)[:,None]*h_#can be removed
			return h, c
		layer_input=T.dot(input,W)+b
		rval,updatas = theano.scan(_step,
					sequences=[mask,layer_input],
					outputs_info=[T.alloc(numpy_floatX(0.),
												n_samples,
												hidden_dim
												),
									T.alloc(numpy_floatX(0.),
											n_samples,
											hidden_dim
											)],
									n_steps=n_steps,go_backwards=go_backwards)
		self.h=	rval[0]
		self.params={self.name+"_W":W,self.name+"_U":U,self.name+"_b":b}
		if weight==None:
			lstm_out = (self.h*self.mask[:,:,None]).sum(axis=0)
			self.mean_pool_out = lstm_out/(self.mask.sum(axis=0)[:,None])
		else:
			self.params[self.name+"_u"]=u
			temp=T.dot(self.h, u)
			e_x=T.exp(temp-temp.max(axis=0,keepdims=True))
			e_x_mask=e_x*self.mask
			attention=e_x_mask/e_x_mask.sum(axis=0,keepdims=True)
			self.weighted_pool_out=(self.h*attention[:,:,None]).sum(axis=0)
		
		

##################################################################################################
#										dropout layer
###################################################################################################
class dropout_layer(object):
	def __init__(self,drop_input,use_noise,trng):
		drop_output=T.switch(use_noise,
					(drop_input*
					trng.binomial(drop_input.shape,
							p=0.5,n=1,
					dtype=drop_input.dtype)),
					drop_input*0.5)
		self.output=drop_output
##################################################################################################
#										convolution layer
###################################################################################################
class conv_layer(object):
	name="conv_layer"
	count=0
	def __init__(self,rng,input,filters_shape,input_shape,border_mode='valid'):
		conv_layer.count=conv_layer.count+1
		self.name=pp(conv_layer.name,str(conv_layer.count))
		input=input.dimshuffle(1,'x',0,2)
		w_bound=np.sqrt(filters_shape[1]*filters_shape[2]*filters_shape[0])
		W= np.asarray(
                rng.uniform(
                    low=-w_bound,
                    high=w_bound,
                    size=filters_shape
                ),
                dtype=theano.config.floatX
            )
		W = theano.shared(value=W, name='W',borrow=True)
		b = theano.shared(
            value=np.zeros(
                (filters_shape[0],),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
		conv_out =conv2d(input, W, filter_shape = filters_shape, input_shape=input_shape,border_mode=border_mode)
		output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
		output=T.flatten(output,ndim=3)
		self.output=output.dimshuffle(2,0,1)
		self.params={self.name+"_W":W, self.name+"_b":b}
		self.shape=filters_shape
	def get_mask(self,mask):
		return mask[:T.shape(self.output)[0],:]
class conv_layer_temp(object):
	name="conv_layer_temp"
	count=0
	def __init__(self,rng,input,filters_shape,input_shape,border_mode='half',pooling=""):
		conv_layer.count=conv_layer.count+1
		self.name=pp(conv_layer.name,str(conv_layer.count))
		self.rng=rng
		input=input.dimshuffle(1,2,0,'x')
		w_bound=np.sqrt(filters_shape[1]*filters_shape[2]*filters_shape[0])
		
		W=self.uniform(w_bound,size=filters_shape) 
		W = theano.shared(value=W, name='W',borrow=True)
		
		b = theano.shared(value=self.uniform(0.,(filters_shape[0],)),borrow=True)
		
		W_gate=self.uniform(w_bound,size=filters_shape) 
		W_gate = theano.shared(value=W_gate, name='W_gate',borrow=True)
		
		b_gate=theano.shared(value=self.uniform(0.,(filters_shape[0],)),borrow=True)
		
		conv_out =conv2d(input, W, filter_shape=filters_shape, input_shape=input_shape)
		conv_out_gate=conv2d(input,W_gate,filter_shape=filters_shape,input_shape=input_shape)
		output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))*T.tanh(conv_out_gate+b_gate.dimshuffle('x', 0, 'x', 'x'))
		
		if pooling!="":
			output=T.signal.pool.pool_2d(output,ws=(2,1),ignore_border=True,mode=pooling)
		output=T.flatten(output,ndim=3)
		self.output=output.dimshuffle(2,0,1)
		self.params={pp(self.name,"W"):W, pp(self.name,"b"):b, pp(self.name,'W_gate'):W_gate,pp(self.name,"b_gate"):b_gate}
		self.shape=filters_shape
	def uniform(self,stdev, size):
		"""uniform distribution with the given stdev and size"""
		return self.rng.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)
	def get_mask(self,mask):
		return mask[:T.shape(self.output)[0],:]
##################################################################################################
#										wavenet layer
###################################################################################################
class wavenet_layer(object):
	"""
		remove the causal constrain, so the filter need't pad zeros.
	"""
	name="wavenet_layer"
	count=0
	def __init__(self, input, rng, filter_width,
				in_channels, residual_channels=64, dilation_channels=48, skip_channels=32,out_channels=32,
				dilation_levels=4):
		wavenet_layer.count=wavenet_layer.count+1
		self.name=pp(wavenet_layer.name,str(wavenet_layer.count))
		self.rng=rng
		self.filter_width=filter_width
		self.in_channels=in_channels
		self.residual_channels=residual_channels
		self.dilation_channels=dilation_channels
		self.skip_channels=skip_channels
		self.out_channels=out_channels
		self.params={}
		input=input.dimshuffle(1,2,0,'x')
		output=self.dilatedconv1d(in_channels,residual_channels,self.filter_width,input,1,pp(self.name,"causal_conv"))
		skip_list=[]
		for i in range(dilation_levels):
			d =np.left_shift(2, i)
			skip,residual=self.dilatedconv_stack(output, self.filter_width,d,
									pp(self.name,"dilatedconv_stack")+str(d), i==(dilation_levels-1))
			output=T.nnet.relu(residual)
			skip_list.append(skip)
		sum_list=sum(skip_list)
		#sum_list=T.concatenate(skip_list,axis=1)
		sum_list=T.nnet.relu(sum_list)
		output=self.dilatedconv1d(self.skip_channels,out_channels,1,sum_list,1,pp(self.name,"output_conv"))
		#shape(batchsize, channel, H, W)--->>>> shape(H,batchsize,channels,W)
		output=output.dimshuffle(2, 0, 1, 3)
		#shape(H,batchsize,channels,W)----->>>> shape(H,batchsize,channels)
		output=T.flatten(output,ndim=3)
		self.output=output
	#@staticmethod
	def uniform(self,stdev, size):
		"""uniform distribution with the given stdev and size"""
		return self.rng.uniform(
            low=-stdev * np.sqrt(3),
            high=stdev * np.sqrt(3),
            size=size
        ).astype(theano.config.floatX)
		
	def dilatedconv1d(self,input_dim, output_dim, filter_size, inputs, dilation,name):
		filter_weight=self.uniform(1/np.sqrt(input_dim*filter_size),(output_dim,input_dim,filter_size,1))
		filter_weight=theano.shared(value=filter_weight,borrow=True)
		filter_bais=np.zeros(shape=(output_dim,)).astype(theano.config.floatX)
		filter_bais=theano.shared(value=filter_bais,borrow=True)
		result = T.nnet.conv2d(inputs, filter_weight, border_mode='half', filter_flip=False, filter_dilation=(dilation, 1))
		result+=filter_bais[None,:,None,None]
		self.params.update({pp(name,"filter_weight"):filter_weight,pp(name,"filter_bais"):filter_bais})
		return result
	def dilatedconv_stack(self,input,filter_size,dilation,name,top_stack=False):
		shape=(self.dilation_channels, self.residual_channels, filter_size, 1)
		filter_weight=self.uniform(1/np.sqrt(self.residual_channels*filter_size),shape)
		filter_weight=theano.shared(value=filter_weight,borrow=True)
		filter_bais=np.zeros(shape=(self.dilation_channels,)).astype(theano.config.floatX)
		filter_bais=theano.shared(value=filter_bais,borrow=True)
		
		shape=(self.dilation_channels, self.residual_channels, filter_size, 1)
		gate_weight=self.uniform(1/np.sqrt(self.residual_channels*filter_size), shape)
		gate_weight=theano.shared(value=gate_weight, borrow=True)
		gate_bais=np.zeros(shape=(self.dilation_channels,)).astype(theano.config.floatX)
		gate_bais=theano.shared(value=gate_bais, borrow=True)
		
		shape=(self.skip_channels,self.dilation_channels,1,1)
		skip_weight=self.uniform(1./np.sqrt(self.skip_channels*self.dilation_channels),shape)
		skip_weight=theano.shared(value=skip_weight,borrow=True)
		skip_bais=np.zeros(shape=(self.skip_channels,)).astype(theano.config.floatX)
		skip_bais=theano.shared(value=skip_bais,borrow=True)
		
		shape=(self.residual_channels, self.dilation_channels, 1, 1)
		transformed_weight=self.uniform(1./np.sqrt(self.residual_channels*self.dilation_channels),shape)
		transformed_weight=theano.shared(value=transformed_weight, borrow=True)
		transformed_bais=np.zeros(shape=(self.residual_channels,)).astype(theano.config.floatX)
		transformed_bais=theano.shared(value=transformed_bais, borrow=True)
		 
		conv_filter=T.nnet.conv2d(input,filter_weight,border_mode='half', filter_flip=False, filter_dilation=(dilation, 1))
		conv_gate=T.nnet.conv2d(input, gate_weight, border_mode='half', filter_flip=False, filter_dilation=(dilation, 1))
		out=T.tanh(conv_filter+filter_bais[None,:,None,None])*T.nnet.sigmoid(conv_gate+gate_bais[None,:,None,None])
		
		transformed = T.nnet.conv2d(out, transformed_weight, border_mode='half', filter_flip=False, filter_dilation=(1, 1))
		transformed=transformed+transformed_bais[None,:,None,None]
		
		skip = T.nnet.conv2d(out, skip_weight, border_mode='half', filter_flip=False, filter_dilation=(1, 1))
		skip=skip+skip_bais[None,:,None,None]
		self.params.update({pp(name,"filter_weight"):filter_weight,
							pp(name,"filter_bais"):filter_bais,
							pp(name,"gate_weight"):gate_weight,
							pp(name,"gate_bais"):gate_bais,
							pp(name,"skip_weight"):skip_weight,
							pp(name,"skip_bais"):skip_bais
							})
		if top_stack==False:
			self.params.update({pp(name,"transformed_weight"):transformed_weight,
							pp(name,"transformed_bais"):transformed_bais})
		return skip,transformed#+input
##################################################################################################
#										wavenet layer
###################################################################################################
 
		
		
		
		