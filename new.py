import os
import csv
import librosa
import sys
import numpy as np
import pandas as pd
import glob
import random
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
RNG_SEED=123
class Constants:
    def __init__(self):
        real_path = os.path.dirname(os.path.realpath(__file__))
        self.available_emotions = np.array(['ang', 'exc', 'neu', 'sad'])
        self.path_to_data = real_path + '\\..\\..\\..\\..\\..\\dataset\\IEMOCAP_full_release\\'
        self.path_to_features = real_path + "/../../data/features/"
        self.sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
        self.conf_matrix_prefix = 'iemocap'
        self.framerate = 16000
        self.types = {1: np.int8, 2: np.int16, 4: np.int32}
    
    def __str__(self):
        def display(objects, positions):
            line = ''
            for i in range(len(objects)):
                line += str(objects[i])
                line = line[:positions[i]]
                line += ' ' * (positions[i] - len(line))
            return line
        
        line_length = 100
        ans = '-' * line_length
        members = [attr for attr in dir(self) if not callable(attr) and not attr.startswith("__")]
        for field in members:
            objects = [field, getattr(self, field)]
            positions = [30, 100]
            ans += "\n" + display(objects, positions)
        ans += "\n" + '-' * line_length
        return ans
########################################################################################################################
#                                                 data reading                                                         #
########################################################################################################################


def get_audio(path_to_wav, filename, params=Constants()):
    y,sr =librosa.load(path_to_wav + filename,sr=None,mono=True)
    return y,sr


def get_transcriptions(path_to_transcriptions, filename, params=Constants()):
    f = open(path_to_transcriptions + filename, 'r').read()
    f = np.array(f.split('\n'))
    transcription = {}
    for i in range(len(f) - 1):
        g = f[i]
        i1 = g.find(': ')
        i0 = g.find(' [')
        ind_id = g[:i0]
        ind_ts = g[i1+2:]
        transcription[ind_id] = ind_ts
    return transcription


def get_emotions(path_to_emotions, filename, params=Constants()):
    f = open(path_to_emotions + filename, 'r').read()
    f = np.array(f.split('\n'))
    idx = f == ''
    idx_n = np.arange(len(f))[idx]
    emotion = []
    for i in range(len(idx_n) - 2):
        g = f[idx_n[i]+1:idx_n[i+1]]
        head = g[0]
        i0 = head.find(' - ')
        start_time = float(head[head.find('[') + 1:head.find(' - ')])
        end_time = float(head[head.find(' - ') + 3:head.find(']')])
        actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:
                        head.find(filename[:-4]) + len(filename[:-4]) + 5]
        emo = head[head.find('\t[') - 3:head.find('\t[')]
        vad = head[head.find('\t[') + 1:]
        gender=1 if actor_id[0]=="M" else 0
        v = float(vad[1:7])
        a = float(vad[9:15])
        d = float(vad[17:23])
        
        j = 1
        emos = []
        while g[j][0] == "C":
            head = g[j]
            start_idx = head.find("\t") + 1
            evoluator_emo = []
            idx = head.find(";", start_idx)
            while idx != -1:
                evoluator_emo.append(head[start_idx:idx].strip().lower()[:3])
                start_idx = idx + 1
                idx = head.find(";", start_idx)
            emos.append(evoluator_emo)
            j += 1

        emotion.append({'start': start_time,
                        'end': end_time,
                        'id': filename[:-4] + '_' + actor_id,
						'gender':gender,
                        'v': v,
                        'a': a,
                        'd': d,
                        'emotion': emo,
                        'emo_evo': emos})
    return emotion


def split_wav(wav, emotions, params=Constants()):
    y,sr= wav
    frames = []
    for ie, e in enumerate(emotions):
        start = e['start']
        end = e['end']
        e['signal'] = y[int(start * sr):int(end * sr)]
        frames.append(e['signal'])
    return frames
def get_field(data, key):
    return np.array([e[key] for e in data])

def read_iemocap_data(params=Constants()):
    data = []
    for session in params.sessions:
        path_to_wav = params.path_to_data + session + "\\dialog\\wav\\"
        path_to_emotions = params.path_to_data + session + '\\dialog\\EmoEvaluation\\'
        path_to_transcriptions = params.path_to_data + session + '\\dialog\\transcriptions\\'

        files = os.listdir(path_to_wav)
        files = [f[:-4] for f in files if f.endswith(".wav")]
        for f in files:           
            wav = get_audio(path_to_wav, f + '.wav')
            transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
            emotions = get_emotions(path_to_emotions, f + '.txt')
            sample = split_wav(wav, emotions)

            for ie, e in enumerate(emotions):
                e['transcription'] = transcriptions[e['id']]
                if e['emotion'] in params.available_emotions:
                    data.append(e)
    sort_key = get_field(data, "id")
    return np.array(data)[np.argsort(sort_key)]
def feature_extraction(path='emotion.npy',params=Constants()):
	sr=16000
	data=read_iemocap_data()
	X=[]
	Y=[]
	G=[]
	for ie,e in tqdm(enumerate(data)):
		x=librosa.feature.melspectrogram(y=e['signal'],sr=sr,n_fft=512,hop_length=256,n_mels=64).T##shape=(t,n_mels)
		y=e['emotion']
		g=e['gender']
		X.append(x)
		Y.append(y) 
		G.append(g)
	X=np.array(X)
	Y=label_binarize(Y,params.available_emotions).argmax(axis=1)
	G=np.array(G)
	np.savez('feature',x=X,y=Y,g=G)
	
class datagenerate(object):
	def __init__(self,batchsize=32,shuffle=True,
					portion2train=0.75,
					path2data="feature.npz",RNG_SEED=123):
		file=np.load(path2data)
		X=file['x']
		Y=file['y']
		G=file['g']
		if shuffle==True:
			items=zip(X,Y,G)
			random.Random(RNG_SEED).shuffle(items)
			X,Y,G=zip(*items)
		samples2train=int(portion2train*len(Y))
		self.train_x=X[:samples2train]
		self.train_y=Y[:samples2train]
		self.train_g=G[:samples2train]
		self.test_x=X[samples2train:]
		self.test_y=Y[samples2train:]
		self.test_g=G[samples2train:]
	def padding(self,batch):
		lengths=[it.shape[0] for it in batch]
		maxlen=max(lengths)
		minibatch=np.zeros((maxlen,len(batch), batch[0].shape[1]),dtype=np.float32)#shape=(timestep,batchsize,featdim)
		mask=np.zeros((maxlen,len(batch)),dtype=np.float32)#shape=(timestep,batchsize)
		for i in range(len(lengths)):
			feat=batch[i]
			minibatch[:feat.shape[0],i,:]=10*feat
			mask[:feat.shape[0],i]=1
		return minibatch,mask
		
	def iterate(self,features,labels,gender,batchsize=32,max_iters=None):
		len_samples=len(labels)
		if max_iters==None:
			max_iters=np.ceil(len_samples/batchsize)
		max_iters=int(max_iters)
		start=0
		end=0
		for iter in range(max_iters):
			start=end
			end=min(end+batchsize,len_samples)
			yield self.padding(features[start:end]),np.array(labels[start:end],dtype=np.int32), np.array(gender[start:end],dtype=np.int32)
			
	def train_iterate(self,batchsize=32,sort_by_duration=False,shuffle=True):
		if sort_by_duration and shuffle:
			shuffle=False
		if sort_by_duration:
			duration=[it.shape[0] for it in self.train_x]
			duration, self.train_x, self.train_y=zip(*sorted(zip(duration,self.train_x,self.train_y),key=lambda it:it[0]))
		if shuffle:
			items=zip(self.train_x,self.train_y,self.train_g)
			random.Random(RNG_SEED).shuffle(items)
			self.train_x,self.train_y,self.train_g=zip(*items)
		return self.iterate(self.train_x, self.train_y, self.train_g)
	def test_iterate(self,batchsize=64):
		return self.iterate(self.test_x, self.test_y, self.test_g, batchsize=batchsize)


if __name__=='__main__':
	datagene=datagenerate()
	for (x,mask),y in datagene.test_iterate():
		print x.shape, mask.shape, y.shape