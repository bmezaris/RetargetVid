import os
import gc
import sys
from sys import getsizeof
import math
import time

seed_int = 5
from numpy.random import seed as np_seed
np_seed(seed_int)
from tensorflow import set_random_seed as tf_set_random_seed
tf_set_random_seed(seed_int)

import numpy as np

import keras
from keras import layers
from keras import activations
from keras import models
from keras import optimizers
from keras_applications import imagenet_utils
from keras_applications.imagenet_utils import decode_predictions
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

import cv2

from scipy.signal import argrelextrema
from scipy.signal import find_peaks


def get_fn(s):
	a,b = os.path.split(s)
	c,d = os.path.splitext(b)
	return c
	
def trunc(x, trunc=4):
	dem = float(10**trunc)
	for i in range(len(x)):
		if trunc>0:
			x[i] = float(int(x[i]*dem))/dem
	return x

def mov_avg(x, window=3):
	l = len(x)
	y = np.zeros((l,1), dtype=float)
	for i, x in enumerate(x,0):
		mov_ave_val = 0
		low_limit = int((window-1)/2)
		high_limit = l-int((window-1)/2)
		if i<low_limit:
			count = 0
			for j in range(0, low_limit):
				count += 1
				mov_ave_val += x[j]
			mov_ave_val /= float(count)
		elif i>=high_limit:
			count = 0
			for j in range(high_limit, l):
				count += 1
				mov_ave_val += x[j]
			mov_ave_val /= float(count)
		else:
			for j in range(i-int((window-1)/2), i+int((window-1)/2)+1):
				mov_ave_val += x[j]
			mov_ave_val /= window
		y[i] = mov_ave_val
	return y
	
def smooth(x, window=3):
	w=np.ones(window,'d')
	y=np.convolve(w/w.sum(), x, mode='same')
	return y
	
def find_extremas(x, order=3):
	lmin = argrelextrema(x, np.less, order=order)[0] 		# local minima
	lmax = argrelextrema(x, np.greater, order=order)[0]		# local maxima
	
	lmin = []
	min = 100000.0
	min_pos = -1
	for j in range(0,lmax[0]):
		if x[j]<min:
			min = x[j]
			min_pos = j
	lmin.append(min_pos)
	
	for i in range(len(lmax)-1):
		min = 100000.0
		min_pos = -1
		# find the minimum between this and the next maxima
		for j in range(lmax[i]+1, lmax[i+1]):
			if x[j]<min:
				min = x[j]
				min_pos = j
		lmin.append(min_pos)
	lmin = np.array(lmin)
		
	lmin = lmin + 1
	lmax = lmax + 1
	
	return lmin, lmax
	
def process_sd_x(x, window=3, order=3, verbose=False):
	l = len(x)
	x_smoothed = smooth(x, window=window)
	mins, maxs = find_extremas(x_smoothed, order=order)
	if verbose:
		print('mins::', len(mins), '-', mins)
		print('maxs::', len(maxs), '-', maxs)
	y = np.zeros(l, dtype=float)
	for k in range(1,len(maxs)):
		y[maxs[k]] = abs(x_smoothed[maxs[k]]-x_smoothed[mins[k-1]]) + abs(x_smoothed[maxs[k]]-x_smoothed[mins[k]])
		if y[maxs[k]]>1.0:
			y[maxs[k]]=1.0
	maxs_t = np.zeros(l, dtype=float)
	for k in maxs:
		maxs_t[k]=x_smoothed[k]
	mins_t = np.zeros(l, dtype=float)
	for k in mins:
		mins_t[k]=x_smoothed[k]
	
	return y, x_smoothed, mins_t, maxs_t
	
def trans_to_boundaries(y, t=0.40):
	bounds = []
	prev = 0
	for i in range(len(y)):
		if y[i]>=t:
			bounds.append([prev+1,i])
			prev=i
	bounds.append([prev+1, len(y)])
	return bounds
	
def trans_to_list(y, t=0.40):
	l = []
	for i in range(len(y)):
		if y[i]>=t:
			l.append(i)
			prev=i
	l.append(len(y))
	return l
	

if __name__ == "__main__":
	epsilon = 1e-7
	frames_to_process = 10
	batch_size=10
	print_more_info = False


	print(' Scanning test files...')
	root_dir = os.path.join('test', 'video_rai')
	video_files = []
	frameCounts = []
	lengths = []
	total_length = 0
	for root, directories, filenames in os.walk(root_dir):
		for filename in filenames: 
			if filename.endswith('mp4'):
				video_files.append(os.path.join(root,filename))
				cap = cv2.VideoCapture(video_files[-1])
				frameCounts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
				fps = cap.get(cv2.CAP_PROP_FPS)
				lengths.append(frameCounts[-1]/fps)
				total_length += lengths[-1]
				print(video_files[-1],'-',frameCounts[-1])


	print('\n Scanning model files...')
	models_dir = os.path.join('model', 'snapshots')
	models = []
	for m in os.listdir(models_dir):
		if m.endswith('.hdf5'):
			models.append(os.path.join(models_dir,m))
			print(models[-1])

	
	print('\n Predicting...')	
	for mi,m in enumerate(models):
		model_name = m.split('fsd_')[1]
		model_name = m.split('.hdf5')[0]

		if model_name[model_name.find('cm=')+3]=='c':
			class_mode = 'categorical'
			loss = 'categorical_crossentropy'
		else:
			class_mode = 'binary'
			loss = 'binary_crossentropy'
			
		lr = float(model_name[model_name.find('ls=')+3:model_name.find('-sp=')])

		samplewise_pre = bool(int(model_name[model_name.find('sp=')+3]))
		featurewise_pre = bool(int(model_name[model_name.find('fp=')+3]))

		if model_name[model_name.find('opt=')+4:model_name.find('opt=')+7]=='sgd':
			opt = optimizers.SGD(lr=lr,
						momentum=0.0, 
						decay=0.0, 
						nesterov=False)
		else:
			opt = optimizers.Adam(lr=lr, 
						beta_1=0.9, 
						beta_2=0.999, 
						epsilon=None, 
						decay=0.0, 
						amsgrad=False)
		epoch = int(model_name[model_name.find('_e')+2:])

		print('%2d/%2d model... (cm=%s, s_pre=%d, f_pre=%d, lr=%.5f, epoch=%d)' % (mi+1, len(models), class_mode, samplewise_pre, featurewise_pre, lr, epoch))
		fsd = load_model(m)
		fsd.compile(optimizer=opt, loss=loss,	metrics=['accuracy'])

		row_axis = 0
		col_axis = 1
		channel_axis = 2

		pred_t = 0
		for i,vid in enumerate(video_files):
			c = -1
			start_time = time.time()
			cap = cv2.VideoCapture(vid)
			if print_more_info:
				print('%d\%d %s (%d), ' % (i+1, len(video_files), get_fn(vid), frameCounts[i]), end='')
			
			f = [None] * frameCounts[i]
			for j in range(frameCounts[i]):
				ret, frame = cap.read()
				x = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
				x = cv2.resize(x, (64,64)).astype(np.float)
				x /= 255
				
				if samplewise_pre:
					mean = np.mean(x, axis=(channel_axis, row_axis, col_axis))
					std = np.std(x, axis=(channel_axis, row_axis, col_axis))
					x -= mean
					x /= (std + epsilon)
				f[j] = x
				
			y_pred = [0] * frameCounts[i]
			for j in range(0,frameCounts[i]-frames_to_process,batch_size):
				Xb = np.zeros((batch_size, frames_to_process, 64, 64, 3), dtype=float)
				for b in range(batch_size):
					if print_more_info:
						print('batch:', (b+j), 'inds: ', end ='')
					for t in range(frames_to_process):
						if print_more_info:
							print('%5d ' % (j+b+t), end='')
						if (j+b+t)<frameCounts[i]:
							Xb[b,t,:,:,:] = f[j+b+t]
						else:
							Xb[b,t,:,:,:] = Xb[-1,-1,:,:,:]
						
					if print_more_info:
						print('')
				
				if print_more_info:			
					print('input shape=', Xb.shape)
				pred = fsd.predict(Xb, batch_size=batch_size)
				pred = np.squeeze(pred)
				if class_mode=='categorical':
					pred = pred[:,1]
				if print_more_info:
					print('output shape=', pred.shape)
				y_pred[j:j+batch_size] = pred
			
			with open(os.path.join(root_dir, get_fn(vid)+'_pred-'+'cm=%s,s_pre=%d,e=%d'%(class_mode, samplewise_pre, epoch)+'.txt'), 'w') as f:
				for j in range(frameCounts[i]):
					f.write('%.3f\n'%y_pred[j])



	
	
