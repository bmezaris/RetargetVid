# basic imports
import copy
import math
import os
import pathlib
import sys
import numpy as np
import pickle
import gc

# OpenCV for reading images from disk
import cv2

# for saving demo
import ffmpeg

# utils for clustering
import hdbscan

# imutils for fast video reading
from imutils.video import FileVideoStream
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans

# for statistics calculation
import statistics

# SciPy's interpolation
from scipy import interpolate

# SciPy's butterworth lowpass filters
from scipy import signal

# get path that the current script file is in
root_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
temp_path = None
temp_path = os.path.join(root_path, 'temp')

### 3rd party libs loading
# DCNN-based shot segmentation using TransNet
import tensorflow as tf
tfv = str(tf.__version__)
if tfv[0]=='1':
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	transnetv1_full_path = os.path.join(root_path, '3rd_party_libs', 'transnetv1')
	print(' (adding path %s)' % transnetv1_full_path)
	sys.path.insert(0, transnetv1_full_path)
	print(' loading transnet v2 model')
	import transnetv1_handler
	stn_params = transnetv1_handler.ShotTransNetParams()
	stn_params.CHECKPOINT_PATH = os.path.join('3rd_party_libs', 'transnetv1', 'shot_trans_net-F16_L3_S2_D256')
	trans_threshold = 0.1
	transnet_model = transnetv1_handler.ShotTransNet(stn_params, session=sess)
	TRANSNET_H = 27
	TRANSNET_W = 48
else:
	transnetv2_full_path = os.path.join(root_path, '3rd_party_libs', 'transnetv2', 'inference')
	print(' (adding path %s)' % transnetv2_full_path)
	sys.path.insert(0, transnetv2_full_path)
	print(' loading transnet v2 model')
	import transnetv2
	trans_threshold = 0.5
	transnet_model = transnetv2.TransNetV2(model_dir=os.path.join(transnetv2_full_path, 'transnetv2-weights'))
	TRANSNET_H = 27
	TRANSNET_W = 48

# DCNN-based Saliency detection using UNISAL
import torch
unisal_full_path = os.path.join(root_path, '3rd_party_libs', 'unisal')
gpu_device = torch.device('cuda:0')
print(' (adding path %s)' % unisal_full_path)
sys.path.insert(0, unisal_full_path)
print(' loading unisal model')
import unisal_handler
unisal_model = unisal_handler.init_unisal_for_images()

# Loess local weighted regression
loess_full_path = os.path.join(root_path, '3rd_party_libs', 'loess')
print(' (adding path %s)' % loess_full_path)
sys.path.insert(0, loess_full_path)
import pyloess

# pyTorch kmeans
kmeans_pytorch_full_path = os.path.join(root_path, '3rd_party_libs', 'kmeans_pytorch')
print(' (adding path %s)' % kmeans_pytorch_full_path)
from kmeans_pytorch import kmeans as ptkm


# Methods to initiate the time dictionary,
# register a time in time dictionary,
# and print the time dictionary
sc_times = {}
def sc_init_time():
	global sc_times
	sc_times = {}

def sc_register_time(t, key_name):
	add_t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
	if key_name in sc_times.keys():
		sc_times[key_name] += add_t
	else:
		sc_times[key_name] = add_t
		
def sc_save_time_override(key_name, t):
	sc_times[key_name] = t

def sc_all_times(vid_dur):
	t_dict =  {}
	sum_t = 0
	sum_p = 0
	for key_name in sc_times.keys():
		if key_name.startswith('_'):
			sum_t += sc_times[key_name]
			sum_p += (sc_times[key_name] / vid_dur) * 100.0
		t_dict[key_name] ='%7.3fs, %6.3f%%' % (sc_times[key_name], 
												(sc_times[key_name]/vid_dur)*100.0)
	t_dict['total'] = '%7.3fs, %6.3f%%' % (sum_t, sum_p)
	return t_dict

def sc_get_time(key_name):
	return sc_times[key_name]



# Initiates the SmartCrop method's parameters to the default settings
def sc_init_crop_params(print_dict=False):
	crop_params = {}

	crop_params['out_ratio'] 		= "4:5"
	crop_params['max_input_d'] 		= 250
	crop_params['skip'] 			= 6
	crop_params['seconds_batch'] 		= 20

	crop_params['resize_factor'] 		= 1.0
	crop_params['op_close'] 		= True
	crop_params['value_bias']		= 1.0 	# bias conversion of image value
							# to 3rd dimension for clustering
									
	crop_params['exit_on_spread_sal'] 	= False # Set to True to resort to padding
							# if mean saliency of a frame is above t_sal
	crop_params['exit_on_low_cvrg'] 	= False # Set to True to resort to padding
							# if the coverage score is above t_cvrg
	
	crop_params['com_km'] 			= True 	# perform kmeans for center of mass,
							# else return position of max val
	
	crop_params['clust_filt'] 		= True
	crop_params['select_sum'] 		= 1  	# if 1, select cluster with max sum, 
							# else select cluster with value
		
	crop_params['use_3d'] 			= False	# use 3d clustering filtering
	
	crop_params['use_fast'] 		= False	# use fast multi-kmeans instead of HDBSCAN
	
	crop_params['min_d_jump']		= 20 	# min pixels distance of a center 
											# jumps to take into consideration
											
	crop_params['min_t_cut'] 		= 1.0 	# length of consecutive empty sal, maps 
											# must be above this to select next cluster
	crop_params['hdbscan_min'] 		= 26

	crop_params['shift_time'] 		= 0

	crop_params['loess_filt'] 		= 1
	crop_params['loess_w_secs'] 		= 2
	crop_params['loess_degree'] 		= 2

	crop_params['lp_filt'] 			= 1
	crop_params['lp_cutoff'] 		= 2
	crop_params['lp_order'] 		= 5

	crop_params['t_sal'] 			= 40  	# max mean saliency to continue (if higher than this -> pad)
	crop_params['t_cvrg'] 			= 0.60  # min coverage to continue (if lower than this -> pad)
	crop_params['t_threshold'] 		= 200
	crop_params['t_border'] 		= -1 	# set to -1 to disable border detection
	
	crop_params['t_cut'] 			= 120 	# if lower than this then a jump over a low saliency area 
											# was made and extra cut will be inserted
	crop_params['no_extra_cuts'] 	= 0

	if print_dict:
		for x in crop_params.keys():
			print(x, ':', crop_params[x])

	return crop_params



# converts an array of probabilities to a structure of shot boundaries
def predictions_to_scenes(predictions: np.ndarray, threshold: float = 0.5):
	predictions = (predictions > threshold).astype(np.uint8)
	scenes = []
	t, t_prev, start = -1, 0, 0
	for i, t in enumerate(predictions):
		if t_prev == 1 and t == 0:
			start = i
		if t_prev == 0 and t == 1 and i != 0:
			scenes.append([start, i])
		t_prev = t
	if t == 0:
		scenes.append([start, i])

	# just fix if all predictions are 1
	if len(scenes) == 0:
		return np.array([[0, len(predictions) - 1]], dtype=np.int32)
	return np.array(scenes, dtype=np.int32)

# Reads a video, performs shot and salienct detection
# return segmentation info and samples saliency mapsd
def read_and_segment_video(video_path, crop_params, verbose=False):
	t = cv2.getTickCount()
	
	# open video, get info and close it
	print(' ingesting %s...' % video_path)
	vcap = cv2.VideoCapture(video_path)
	fr = vcap.get(cv2.CAP_PROP_FPS)
	frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
	w = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
	h = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	vcap.release()
	del vcap
	
	# compute batch size based on frame rate
	batch_size = int(crop_params['seconds_batch']*fr)
	batch_overlap = int(fr)
	
	# compute frames size for saliency detection
	dsr = float(max(w,h)) / crop_params['max_input_d']
	SAL_H = int(h/dsr)
	SAL_W = int(w/dsr)

	# init an array to hold the resized frames for 
	# shot segmentation
	transnet_frames = np.zeros((batch_size+batch_overlap, 
								TRANSNET_H, TRANSNET_W, 3), 
								dtype=np.uint8)
						
	# will hold the transition probs for all and selected frames
	trans_probs = []
	trans_probs_sel = []
								
	# init an array to hold the resized frames for 
	# saliency detection
	frames = np.zeros((batch_size, 
						SAL_H, SAL_W, 3), 
						dtype=np.uint8)
						
	# init video data dictionary
	vid_data = {}
						
	# estimate number of frames taking into account skipped onew
	fcs = 0
	skip = crop_params['skip']
	for i in range(frame_count):
		if (i%skip==0) or (i==0) or i==(frame_count-1):
			fcs += 1
	fcs += int(frame_count*0.1)
	if verbose:
		print(' (estimating %d sal. maps)' % fcs)
				
	# init an array to hold saliency maps
	vid_data['smaps'] = np.zeros((SAL_H, SAL_W, fcs), 
								dtype=np.uint8)
					
	# init a list to hold true indices of processed frames
	true_inds = [] # in=sampled frame index, out=respective true frame index
	map2orig = [] # in=true frame index, out=respective sampled frame index
	
	# init current batch size and batch count
	bc = 0
	bsi = -1
	total_process_ind = -1
	
	# setup fast video reader
	fvs = FileVideoStream(video_path).start()
	iii = -1
	
	# register read_init time
	sc_register_time(t, 'read_init')
	
	# loop through video
	bail_out = False
	after_shot_change = False
	while fvs.more():
		# set timer for "read" time
		t = cv2.getTickCount()
		
		# show header and try to load frame
		if (iii%9==0) or (iii>frame_count-50):
			print('\r reading %d/%d ' % (iii+1, frame_count), end='', flush=True)
		frame = fvs.read()
		if frame is None:
			bail_out = True
		else:
			
			# increment indices
			iii += 1
			bsi += 1

			# convert frame to RGB
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			# register frame for shot segmentation
			transnet_frames[batch_overlap+bsi,:,:,:] = cv2.resize(frame, 
											(TRANSNET_W, TRANSNET_H),
											interpolation=cv2.INTER_LINEAR)
											 
			# register frame for shot segmentation
			frames[bsi,:,:,:] = cv2.resize(frame, 
									(SAL_W, SAL_H),
									interpolation=cv2.INTER_LINEAR)
									
		# compute current batch	
		cur_batch_len = bsi+1
		
		# register read time
		sc_register_time(t, '_read')
			
		# process frames if we have a complete or the last batch
		if cur_batch_len==batch_size or bail_out:
			
			# if new batch is empty break
			if cur_batch_len==0:
				break
				
			# set timer for "read_shot_det" time
			t = cv2.getTickCount()
		
			# if not 1st batch, append batch overlap to shot detection frames
			if bc>0:
				prev_batch_len = prev_transnet_frames.shape[0]
				for i in range(batch_overlap):
					transnet_frames[i,:,:,:] = prev_transnet_frames[prev_batch_len-batch_overlap+i,:,:,:]
			prev_transnet_frames =  np.copy(transnet_frames)
				
			# comput start and end save indices for shot detection
			si = bc*batch_size
			ei = si+cur_batch_len
			
			# print info
			print('\r reading %d/%d ' % (iii + 1, frame_count), end='', flush=True)
			print('\n processing batch %d (%d->%d)...' % (bc,si,ei))
				
			# shot detection - copy non-overlapping probs
			temp = transnet_model.predict_video(transnet_frames)
			for i in range(cur_batch_len):
				trans_probs.append(temp[batch_overlap+i])
			
			# zero transnet_frames array
			transnet_frames.fill(0)

			# gather selected (non-skipped) frames
			process_ind = -1

			for i in range(cur_batch_len):
				# check if we must process this frame
				# force if:
				# - force if frame after shot cut)
				# - shot change
				# - 1st frame of 1st batch
				# - last frame
				if ((si+i==true_inds[-1]+skip) if len(true_inds)>0 else True) or \
				   (after_shot_change) or \
				   (si+i==frame_count-1):
					total_process_ind += 1
					process_ind += 1
					frames[process_ind] = frames[i]
					true_inds.append(si+i)
					trans_probs_sel.append(trans_probs[si+i])
				if after_shot_change:
					after_shot_change=False
				after_shot_change = (trans_probs[si+i]>trans_threshold)
				
				# register map index from
				map2orig.append(total_process_ind)
				
			# register shot_det time
			sc_register_time(t, '_read_shot_det')
					
			# set timer for "read_sal_det" time
			t = cv2.getTickCount()		
					
			# comput start and end save indices for shot detection
			ei = len(true_inds)-1
			si = ei-process_ind
			
			# resize saliency maps array if needed
			if ei+1>=vid_data['smaps'].shape[2]:
				if verbose:
					print(' (resizing saliency map array:',vid_data['smaps'].shape, '->', end='')
				vid_data['smaps'].resize((SAL_H, SAL_W, ei+100))
				if verbose:
					print(vid_data['smaps'].shape, ')')
				
			# saliency detection
			vid_data['smaps'][:,:,si:ei] = unisal_handler.predictions_from_memory_nuint8_np(unisal_model,
														frames[:process_ind,:,:,:], [], '')
				
			# zero frames array
			frames.fill(0)
			
			# register sal_det time
			sc_register_time(t, '_read_sal_det')
				
			# increment batch count
			bc += 1
			
			# reset current batch size
			bsi = -1
			
			# exit loop if this is the last batch we process
			if bail_out:
				break
			
	# get true frame count (no frames that we successfully read)
	true_frame_count = iii+1
	
	# start timer for "read_tidy" time
	t = cv2.getTickCount()	
				
	# free video reader
	fvs.stop()
	del fvs
	
	# free empty saliency maps at the end of the array
	vid_data['smaps'] = vid_data['smaps'][:,:,:ei+1]
	print(' gathered %d saliency maps...' % vid_data['smaps'].shape[2])

	# infer segments from transition probs
	vid_data['segmentation'] = predictions_to_scenes(np.array(trans_probs), threshold=trans_threshold)
	
	# shot segmentation FIX
	# make sure end of each segment is the start of next
	# (original algorithm leaves black frame off)
	for i in range(vid_data['segmentation'].shape[0]-1):
		vid_data['segmentation'][i][1] = vid_data['segmentation'][i+1][0] - 1
	vid_data['segmentation'][-1][1] = true_frame_count -1
	
	# check for very small segments
	### TODO
	
	# infer segments for selected frames
	vid_data['segmentation_sel'] = np.copy(vid_data['segmentation'])
	for i in range(vid_data['segmentation_sel'].shape[0]):
		for j in range(vid_data['segmentation_sel'].shape[1]):
			vid_data['segmentation_sel'][i][j] = \
				map2orig[vid_data['segmentation_sel'][i][j]]
	
	# clean up
	del trans_probs
	del trans_probs_sel
	
	# fill video data dictionary
	vid_data['true_inds'] = true_inds
	vid_data['inds_to_orig'] = map2orig
	vid_data['fr'] = fr
	vid_data['fc'] = true_frame_count
	vid_data['fc_sel'] = vid_data['smaps'].shape[2]
	vid_data['h_orig'] = h
	vid_data['w_orig'] = w
	vid_data['h_process'] = SAL_H
	vid_data['w_process'] = SAL_W
	
	# register sal_det time
	sc_register_time(t, 'read_tidy')

	# print info
	if verbose:

		print(' %-24s: %d'%('segmentation end', vid_data['segmentation'][-1][-1]))
				
		print(' %-24s: %.3f'%('frame rate', vid_data['fr']))
		print(' %-24s: %d'%('total frames', vid_data['fc']))
		print(' %-24s: %d'%('true total frames', frame_count))
		print(' %-24s: %d'%('selected frames', vid_data['fc_sel']))
		print(' %-24s: %d'%('true indices len', len(vid_data['true_inds'])))
		print(' %-24s: '%('true indices'))
		print(vid_data['true_inds'])

		print(' %-24s: %d'%('map len', len(vid_data['inds_to_orig'])))
		print(' %-24s: (%dx%d)'%('original dimension',vid_data['h_orig'], 
													  vid_data['w_orig']))
		print(' %-24s: (%dx%d)'%('process dimensions',vid_data['h_process'], 
													  vid_data['w_process']))
		print(' %-24s: (%dx%dx%d)'%('saliency shape',vid_data['smaps'].shape[0], 
													 vid_data['smaps'].shape[1], 
													 vid_data['smaps'].shape[2]))

		print(' %-24s '%('segmentation'))
		print(vid_data['segmentation'])
		print(' %-24s '%('segmentation sel'))
		print(vid_data['segmentation_sel'])
		
	# sanity checks
	all_good = True
	if vid_data['fc']>frame_count:
		print(' Error in sanity check (1)...')
		all_good = False
	if vid_data['fc_sel']!=len(vid_data['true_inds']):
		print(' Error in sanity check (2)...')
		all_good = False
	if vid_data['fc']!=len(vid_data['inds_to_orig']):
		print(' Error in sanity check (3)...')
		all_good = False
	if vid_data['fc_sel']!=vid_data['smaps'].shape[2]:
		print(' Error in sanity check (4)...')
		all_good = False
	if vid_data['segmentation'][-1][-1]!=vid_data['fc']-1:
		print(' Error in sanity check (5)...')
		all_good = False
	if vid_data['segmentation_sel'][-1][-1]!=vid_data['fc_sel']-1:
		print(' Error in sanity check (6)...')
		all_good = False
	if vid_data['inds_to_orig'][-1]!=vid_data['fc_sel']-1:
		print(' Error in sanity check (7)...')
		all_good = False
	if all_good and verbose:
		print(' (sanity checks passed)')
	if not all_good:
		input('...')
		
	# save times on video data dictionary
	# to be available when re-loading video data
	vid_data['times'] = {}
	vid_data['times']['read_init'] 		= sc_get_time('read_init')
	vid_data['times']['_read'] 			= sc_get_time('_read')
	vid_data['times']['_read_shot_det'] = sc_get_time('_read_shot_det')
	vid_data['times']['_read_sal_det'] 	= sc_get_time('_read_sal_det')
	vid_data['times']['read_tidy'] 		= sc_get_time('read_tidy')
			
	return vid_data

# Checks for blank borders in the video frames
# computes the dimensions to cut
def sc_border_detection(crop_params, vid_data, verbose=False):
	
	if crop_params['t_border']==-1:
		# if border detection is off set all border to zero
		vid_data['border_t'] = 0
		vid_data['border_b'] = 0
		vid_data['border_l'] = 0
		vid_data['border_r'] = 0
		return vid_data
		
	# alias process dimensions for quick reference
	h = vid_data['h_process']
	w = vid_data['w_process']
	ho = vid_data['h_orig']
	wo = vid_data['w_orig']
	
	# get min across time (3d to 2d)
	sal_max = np.max(vid_data['smaps'], axis=2)

	# get max across rows (2d to column)
	f_col = np.max(sal_max, axis=1)
	
	# get max acroos cols (2d to row)
	f_row = np.max(sal_max, axis=0)
	
	# print info
	if verbose:
		np.set_printoptions(edgeitems=10)
		print('   column max sal (top-botom borders) (%d):\n'%len(f_col), f_col)
		print('   row max sal (left and right borders) (%d):\n'%len(f_row), f_row)
		np.set_printoptions(edgeitems=3)

	# compute top and bottom borders
	t = 0
	for i in range(h):
		if f_col[i] > crop_params['t_border']:
			break
		t += 1
	b = 0
	for i in range(h):
		if f_col[-(i + 1)] > crop_params['t_border']:
			break
		b += 1

	# compute left and right borders
	l = 0
	for i in range(w):
		if f_row[i] > crop_params['t_border']:
			break
		l += 1
	r = 0
	for i in range(w):
		if f_row[-(i + 1)] > crop_params['t_border']:
			break
		r += 1

	# register limited borders
	vid_data['border_t'] = min(t, int(h*0.25))
	vid_data['border_b'] = min(b, int(h*0.25))
	vid_data['border_l'] = min(l, int(w*0.25))
	vid_data['border_r'] = min(r, int(w*0.25))
	
	# borders will be use when considering original dimensions
	# scale back to original
	vid_data['border_t'] = int((ho/h)*vid_data['border_t'])
	vid_data['border_b'] = int((ho/h)*vid_data['border_b'])
	vid_data['border_l'] = int((wo/w)*vid_data['border_l'])
	vid_data['border_r'] = int((wo/w)*vid_data['border_r'])
		
	if verbose:
		print(' %-24s: (%dx%d) t=%d,b=%d,l=%d,r=%d' % ('border detection',
										ho,wo,
										vid_data['border_t'] ,
										vid_data['border_b'] ,
										vid_data['border_l'] ,
										vid_data['border_r'] ))
	return vid_data

# Computes the IoU of two rectangles
def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


# performs a fast pyTorch kmeans implementation that utilizes GPU
# the clustering is performed over various k and 
# choose the result that maximizes the inter cluster distances
# INCOMPLETE IMPLEMENTATION
def sc_multi_kmeans(X,verbose=False):
	verbose_overide = False
	use_inertia = True
	
	# upload X to GPU memory
	t = cv2.getTickCount()
	Xgpu = torch.from_numpy(X)
	sc_register_time(t, 'clust_mkm_upload')

	# cluster
	t = cv2.getTickCount()
	inertias = []
	min_inter_dists = []
	labels = []
	centers = []
	distances = []
	ks = [2,3,4,5,6,7,8]
	for k in ks:
		temp_labels,temp_centers,temp_distances = ptkm(X=Xgpu, 
											num_clusters=k, 
											distance='euclidean', 
											device=gpu_device, 
											tqdm_flag=False)
		labels.append(temp_labels)
		centers.append(temp_centers)
		distances.append(temp_distances)
		min_dists,_ = torch.min(temp_distances,dim=1)
		inertias.append(torch.sum(min_dists))
		
		if not use_inertia:
			sorted_dists,_ = torch.sort(temp_distances,dim=1,descending=False)
			min_inter_dists.append(torch.max(sorted_dists[:,1]).item())
	sc_register_time(t, 'clust_mkm_main')
	
	# choose k 
	t = cv2.getTickCount()
	if use_inertia:
		k_choose_inertia = -1
		for i in range(1,len(ks)-1):
			if inertias[i-1]-inertias[i] < inertias[i]-inertias[i+1]:
				k_choose_inertia=i
				break
		# for i in range(len(ks)):
			# prev_dist = -1
			# if i>0:
				# prev_dist= inertias[i-1]-inertias[i]
			# next_dist = -1
			# if i<len(ks)-1:
				# next_dist= inertias[i]-inertias[i+1]
			# if verbose and verbose_overide:
				# s_i = ' '
				# if k_choose_inertia==i:
					# s_i='*'
				# s_m = ' '
				# print(' %3d : %6.3f%s| %6.3f | %6.3f | %6.3f%s' % (ks[i],
																# inertias[i], s_i,
																# prev_dist,
																# next_dist,
																# min_inter_dists[i], s_m))
		labels = labels[k_choose_inertia]
		centers = centers[k_choose_inertia]
		distances = distances[k_choose_inertia]
		n_clusters = ks[k_choose_inertia]

	else:
		k_choose_min_inter_dist = min_inter_dists.index(max(min_inter_dists))
		# for i in range(len(ks)):
			# if verbose and verbose_overide:
				# s_i = ' '
				# if k_choose_inertia==i:
					# s_i='*'
				# s_m = ' '
				# if k_choose_min_inter_dist==i:
					# s_m='*'
				# print(' %3d : %6.3f%s| %6.3f | %6.3f | %6.3f%s' % (ks[i],
																# inertias[i], s_i,
																# prev_dist,
																# next_dist,
																# min_inter_dists[i], s_m))
	
		labels = labels[k_choose_min_inter_dist]
		centers = centers[k_choose_min_inter_dist]
		distances = distances[k_choose_min_inter_dist]
		n_clusters = ks[k_choose_min_inter_dist]
		
	# if verbose and verbose_overide:
		# print(' ...choose %d' % (n_clusters))
	sc_register_time(t, 'clust_mkm_choose')
	
	return labels,centers,distances,n_clusters
	


def sc_calc_dest_size(vid_data, crop_params, verbose=False):
	# compute final dimensions
	c = crop_params['out_ratio'].split(':')
	height_ratio = float(c[0])
	width_ratio = float(c[1])
	# if (width_ratio*orig_width)<(height_ratio*orig_height):
	if height_ratio > width_ratio:
		print(' (portrait-to-landscape conversion preserving width)')
		vid_data['conversion_mode'] = 2
		vid_data['w_final'] = vid_data['w_orig']
		vid_data['h_final'] = int(math.floor((width_ratio / height_ratio) * vid_data['w_orig']))
		vid_data['dim_final'] = vid_data['h_final']
		vid_data['dim_process'] = vid_data['h_process']
	else:
		print(' (landscape-to-portrait preserving height)')
		vid_data['conversion_mode'] = 1
		vid_data['w_final'] = int(math.floor((height_ratio / width_ratio) * vid_data['h_orig']))
		vid_data['h_final'] = vid_data['h_orig']
		vid_data['dim_final'] = vid_data['w_final']
		vid_data['dim_process'] = vid_data['w_process']
		
	if verbose:
		print(' %-24s: (%dx%d)' % ('final dimensions (hxw)',vid_data['h_final'], vid_data['w_final']))
	
	return vid_data

def sc_compute_bb(vid_data, crop_params, verbose=False):
	# alias parameters of quick reference
	frame_h = vid_data['h_orig']
	frame_w = vid_data['w_orig']
	process_h = vid_data['h_process']
	process_w = vid_data['w_process']
	scale_h = float(process_h) / float(frame_h)
	scale_w = float(process_w) / float(frame_w)
	bb_h = vid_data['h_final']
	bb_w = vid_data['w_final']
	bt = vid_data['border_t']
	bb = vid_data['border_b']
	bl = vid_data['border_l']
	br = vid_data['border_r']
	
	# scale center coordinates back to original dimensions
	final_xs = vid_data['dxs']
	final_ys = vid_data['dys']
	for i in range(vid_data['fc']):
		final_xs[i] = int(final_xs[i] / scale_w)
		final_ys[i] = int(final_ys[i] / scale_h)
	
	# compute final crop box dimensions 
	# taking into consideration border detection
	# and register
	fbb_w = bb_w - bl - br
	fbb_h = bb_h - bt - bb
	vid_data['fbb_w'] = fbb_w
	vid_data['fbb_h'] = fbb_h
		
	vid_data['bbs'] = []
	for i in range(vid_data['fc']):
		# compute final bounding box around center coords
		x1 = int(final_xs[i]) - int(math.floor(bb_w / 2.0))
		y1 = int(final_ys[i]) - int(math.floor(bb_h / 2.0))
		x2 = int(final_xs[i]) + int(math.ceil(bb_w / 2.0))
		y2 = int(final_ys[i]) + int(math.ceil(bb_h / 2.0))

		# ensure bounding box has given dimensions
		x1 += (x2 - x1) - fbb_w
		y1 += (y2 - y1) - fbb_h
		
		# ensure bounding box is in frame and borders
		if x1 < bl:
			x1 = bl
			x2 = x1 + fbb_w
		if x2 > frame_w - br:
			x2 = frame_w - br
			x1 = x2 - fbb_w
		if y1 < bt:
			y1 = bt
			y2 = y1 + fbb_h
		if y2 > frame_h - bb:
			y2 = frame_h - bb
			y1 = y1 - fbb_h

		# register bounding boxes
		vid_data['bbs'].append([x1, y1, x2, y2])
		
	return vid_data

def sc_threshold(vid_data, crop_params, copy=False):
	if copy:
		# save a copy of sal. maps to "smaps_orig" 
		# for visualization purposes
		vid_data['smaps_orig'] = np.copy(vid_data['smaps'])
	
	# threshold saliency maps
	vid_data['smaps'][vid_data['smaps'] < crop_params['t_threshold']] = 0
	
	return vid_data



def sc_clustering_filt_3d(hdbs_clusterer, sal_3d_map, CP, VD, fc_offset,
				plots_fn='', verbose=False):
	# will hold the extra cuts that must be inserter
	extra_cuts = []
				
	# if an empty saliency volume was given, return it
	if np.sum(sal_3d_map)==0:
		return sal_3d_map, extra_cuts
		
	# parameters aliases for quick reference
	factor = CP['resize_factor']
	select_sum = CP['select_sum']
	bias = CP['value_bias']
	close = CP['op_close']
	min_t_cut = CP['min_t_cut']
	skip = CP['skip']
	fr = VD['fr']
	
	# compute min count of consecutive empty sal. maps
	# to select next cluster
	min_fc_cut=	int(float(min_t_cut*fr))
	
	# init figure
	t = cv2.getTickCount()
	if plots_fn:
		fig = plt.figure()
	sc_register_time(t, 'clust_plot')
		
	# resize for faster operations
	t = cv2.getTickCount()
	oh = int(sal_3d_map.shape[0])
	ow = int(sal_3d_map.shape[1])
	nh = int(oh / factor)
	nw = int(ow / factor)
	d = sal_3d_map.shape[2]
	if factor!=1.0:
		sal_3d_resized = np.zeros((nh, nw, d), dtype='uint8')
		for i in range(d):
			sal_3d_resized[:,:,i] = cv2.resize(sal_3d_map[:,:,i], 
											(nw, nh), 
											interpolation=cv2.INTER_LINEAR)
	else:
		sal_3d_resized=sal_3d_map
	sc_register_time(t, 'clust_resize_1')
	
	# init & gather points
	t = cv2.getTickCount()
	X = []
	W = []
	inds = np.argwhere(sal_3d_resized > 0)
	for ind in inds:
		X.append([ind[0], ind[1], ind[2], 
				sal_3d_resized[ind[0],ind[1],ind[2]]])
		W.append(X[-1][3])
	l = len(X)
	W = np.array(W).transpose()
	sc_register_time(t, 'clust_gather')
	
	# check dimensions of X
	X = np.array(X)
	if len(X.shape)<2:
		print(sal_3d_map)
		print(X)
		print(X.shape)
		input('...')

	# X rows: x,y,t,val
	# bias 4th row (value of sal map)
	t = cv2.getTickCount()
	max_dim = max(nh, nw)
	X[:, 3] = ((X[:, 3] / np.amax(X[:, 3])) * max_dim * bias)
	X = X.astype(np.uint8)
	sc_register_time(t, 'clust_bias')
	
	if l > CP['hdbscan_min'] + 1:
		# cluster
		# (compute number of clusters excluding outliers)
		labels = hdbs_clusterer.fit_predict(X)
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		sc_register_time(t, 'clust_hdbscan')
		
		# print info
		if verbose:
			print('   saliency volume:', sal_3d_map.shape)
			print('   X:', X.shape)
			print('   d =',d)
			print('   labels len =', len(labels))
			print('   n_clusters =', n_clusters)
					
		# plot before
		t = cv2.getTickCount()
		if plots_fn:
			ax = fig.add_subplot(121, projection='3d')
			sh = ax.scatter(X[:, 1], X[:, 2], X[:, 0], marker="s", c=labels, s=40, cmap="RdBu")
			ax.set_xlabel('x')
			ax.set_ylabel('t')
			ax.set_zlabel('y')
			ax.set_xlim((0, sal_3d_resized.shape[1]))
			ax.set_ylim((0, sal_3d_resized.shape[2]))
			ax.set_zlim((0, sal_3d_resized.shape[0]))
			lg = ax.legend(*sh.legend_elements(), loc='best', title="Clusters")
			ax.grid(True)
		sc_register_time(t, 'clust_plot')
		
		# continue only if more than one clusters
		if n_clusters > 0:

			# calculate weight of each cluster
			t = cv2.getTickCount()
			c_weights = [0] * n_clusters
			for i in range(n_clusters):
				inds = list(np.where(labels == i)[0])
				if select_sum == 1:
					c_weights[i] = np.sum(W[inds])
				else:
					c_weights[i] = np.amax(W[inds])
			max_cl = c_weights.index(max(c_weights))
			sc_register_time(t, 'clust_weighting')

			# calculate per frame the dominant cluster
			t = cv2.getTickCount()
			per_t_cls = [0] * d
			for i in range(d):
				inds = list(np.where(X[:, 2] == i)[0])
				per_t_cls[i] = list(set(labels[inds]))
				if -1 in per_t_cls[i]:
					per_t_cls[i].pop(per_t_cls[i].index(-1))
				cls_weights = [c_weights[x] for x in per_t_cls[i]]
				per_t_cls[i] = [x for _,x in sorted(zip(cls_weights,per_t_cls[i]), reverse=True)]
			sc_register_time(t, 'clust_per_t')

			# compute list of consecutive empty sal maps (cesms)
			t = cv2.getTickCount()
			cesms = []
			started = False
			current_empty_segm = []
			for i in range(d):
				if max_cl not in per_t_cls[i]:
					current_empty_segm.append(i)
					started = True
				if max_cl in per_t_cls[i]:
					if started:
						cesms.append(current_empty_segm)
						current_empty_segm = []
						started=False
			if len(current_empty_segm)>0:
				cesms.append(current_empty_segm)
			sc_register_time(t, 'clust_cesm')
			
			
			# handle empty sal maps
			t = cv2.getTickCount()
			select_cl = [max_cl] * d # init selected c. to max weight c.
			for i in range(len(cesms)):
				
				# compute 
				last_true_ind = VD['true_inds'][fc_offset+cesms[i][-1]]
				first_true_ind = VD['true_inds'][fc_offset+cesms[i][0]]
				true_fc_len = last_true_ind - first_true_ind 
								
				if true_fc_len<min_fc_cut:
					pass # will be hanlded by main method
				else:
					# create a list with 
					# the first of next clusters for each sal map
					# (cluster in per_t_cls are sorted)
					cls = []
					for j in cesms[i]:
						if len(per_t_cls[j])>0:
							cls.append(per_t_cls[j][0])
							
					# if no alternative clusters were found
					# for this segment, continue
					# this will be hanlded by main method
					if len(cls)==0:
						continue
							
					# find the most common cluster
					# it's OK if the cluster is not present in all frames
					# this will be hanlded by main method
					most_common_cl = max(cls, key=cls.count)
					
					# assign the segment's maps the new cluster and 
					# insert a new cut
					extra_cuts.append(cesms[i][0])
					for j in cesms[i]:
						select_cl[j] = most_common_cl
			sc_register_time(t, 'clust_empty_handle')
				
			# filter
			t = cv2.getTickCount()
			sal_3d_resized_backup = np.copy(sal_3d_resized)
			selX = []
			for i in range(len(X)):
				if labels[i] != select_cl[X[i][2]]:
					sal_3d_resized[X[i][0], X[i][1], X[i][2]] = 0
				else:
					selX.append([X[i][0], X[i][1], X[i][2]])
			selX = np.array(selX)
			sc_register_time(t, 'clust_filtering')

			# morphological close
			t = cv2.getTickCount()
			closing_kernel = np.ones((5, 5), np.uint8)
			if close:
				for i in range(d):
					sal_3d_resized[:,:,i] = cv2.morphologyEx(np.squeeze(sal_3d_resized[:,:,i]), 
															cv2.MORPH_CLOSE,
															closing_kernel)
			sc_register_time(t, 'clust_closing')

			# check for empty sal maps
			if verbose:
				print('   - segment (selected cluster %d)' % (max_cl))
				for i in range(d):
					empty_str=' '
					if np.sum(sal_3d_resized[:,:,i])==0:
						empty_str='*'
					print('     %3d %s %-16s (selected=%d)' % (i,empty_str, str(per_t_cls[i]), select_cl[i]))
			
		# plot after	
		t = cv2.getTickCount()	
		if plots_fn:
			ax2 = fig.add_subplot(122, projection='3d')
			sh = ax2.scatter(selX[:, 1], selX[:, 2], selX[:, 0], marker="s", s=40, cmap="RdBu")
			ax2.set_xlabel('x')
			ax2.set_ylabel('t')
			ax2.set_zlabel('y')
			ax2.set_xlim((0, sal_3d_resized.shape[1]))
			ax2.set_ylim((0, sal_3d_resized.shape[2]))
			ax2.set_zlim((0, sal_3d_resized.shape[0]))
			ax2.grid(True)
		sc_register_time(t, 'clust_plot')

	# save figure and clean up
	t = cv2.getTickCount()
	if plots_fn:
		plt.savefig(plots_fn,bbox_inches='tight')
		plt.close(fig)
		del fig
	sc_register_time(t, 'clust_plot')
	
	# resize back
	if factor==1.0:
		sc_register_time(cv2.getTickCount(), 'clust_resize_2')
		return sal_3d_resized, extra_cuts
	t = cv2.getTickCount()
	sal_3d_map.fill(0)
	for i in range(d):
		sal_3d_map[:, :, i] = cv2.resize(sal_3d_resized[:, :, i], (ow, oh), interpolation=cv2.INTER_LINEAR)
	sc_register_time(t, 'clust_resize_2')
	return sal_3d_map, extra_cuts

def sc_clustering_filt_3d_fast(sal_3d_map, 
								CP, VD, fc_offset,
								plots_fn='', verbose=False):
	# will hold the extra cuts that must be inserter
	extra_cuts = []
				
	# if an empty saliency volume was given, return it
	if np.sum(sal_3d_map)==0:
		return sal_3d_map, extra_cuts
	
	# parameters aliases for quick reference
	factor = CP['resize_factor']
	select_sum = CP['select_sum']
	bias = CP['value_bias']
	close = CP['op_close']
	min_t_cut = CP['min_t_cut']
	skip = CP['skip']
	fr = VD['fr']
	
	# compute min count of consecutive empty sal. maps
	# to select next cluster
	min_fc_cut=	int(float(min_t_cut*fr))
	
	# init figure
	t = cv2.getTickCount()
	if plots_fn:
		fig = plt.figure()
	sc_register_time(t, 'clust_plot')
		
	# resize for faster operations
	t = cv2.getTickCount()
	oh = int(sal_3d_map.shape[0])
	ow = int(sal_3d_map.shape[1])
	nh = int(oh / factor)
	nw = int(ow / factor)
	d = sal_3d_map.shape[2]
	if factor!=1.0:
		sal_3d_resized = np.zeros((nh, nw, d), dtype='uint8')
		for i in range(d):
			sal_3d_resized[:,:,i] = cv2.resize(sal_3d_map[:,:,i], 
										(nw, nh), 
										interpolation=cv2.INTER_LINEAR)
	else:
		sal_3d_resized=sal_3d_map
	sc_register_time(t, 'clust_resize_1')
	
	# init & gather points
	t = cv2.getTickCount()
	X = []
	W = []
	inds = np.argwhere(sal_3d_resized > 0)
	for ind in inds:
		X.append([ind[0], ind[1], ind[2], 
				sal_3d_resized[ind[0],ind[1],ind[2]]])
		W.append(X[-1][3])
	l = len(X)
	W = np.array(W).transpose()
	sc_register_time(t, 'clust_gather')
	
	# check dimensions of X
	X = np.array(X)
	if len(X.shape)<2:
		print(sal_3d_map)
		print(X)
		print(X.shape)
		input('...')

	# X rows: x,y,t,val
	# bias 4th row (value of sal map)
	t = cv2.getTickCount()
	max_dim = max(nh, nw)
	X[:, 3] = ((X[:, 3] / np.amax(X[:, 3])) * max_dim * bias)
	X = X.astype(np.uint8)
	sc_register_time(t, 'clust_bias')
	
	# clustering: call multi-pytorch-kmeans	
	labels,centers,distances,n_clusters=sc_multi_kmeans(X,verbose=verbose)
		
	# print info
	if verbose:
		print('   saliency volume:', sal_3d_map.shape)
		print('   X:', X.shape)
		print('   d =',d)
		print('   labels len =', len(labels))
		print('   n_clusters =', n_clusters)
				
	# plot before
	t = cv2.getTickCount()
	if plots_fn:
		ax = fig.add_subplot(121, projection='3d')
		sh = ax.scatter(X[:, 1], X[:, 2], X[:, 0], marker="s", c=labels, s=40, cmap="RdBu")
		ax.set_xlabel('x')
		ax.set_ylabel('t')
		ax.set_zlabel('y')
		ax.set_xlim((0, sal_3d_resized.shape[1]))
		ax.set_ylim((0, sal_3d_resized.shape[2]))
		ax.set_zlim((0, sal_3d_resized.shape[0]))
		lg = ax.legend(*sh.legend_elements(), loc='best', title="Clusters")
		ax.grid(True)
	sc_register_time(t, 'clust_plot')
	
	# continue only if more than one clusters
	if n_clusters > 0:

		# calculate weight of each cluster
		t = cv2.getTickCount()
		c_weights = [0] * n_clusters
		for i in range(n_clusters):
			inds = list(np.where(labels == i)[0])
			if select_sum == 1:
				c_weights[i] = np.sum(W[inds])
			else:
				c_weights[i] = np.amax(W[inds])
		max_cl = c_weights.index(max(c_weights))
		sc_register_time(t, 'clust_weighting')

		# calculate per frame the dominant cluster
		t = cv2.getTickCount()
		per_t_cls = [0] * d
		for i in range(d):
			inds = list(np.where(X[:, 2] == i)[0])
			per_t_cls[i] = list(set(labels[inds]))
			if -1 in per_t_cls[i]:
				per_t_cls[i].pop(per_t_cls[i].index(-1))
			cls_weights = [c_weights[x] for x in per_t_cls[i]]
			per_t_cls[i] = [x for _,x in sorted(zip(cls_weights,per_t_cls[i]), reverse=True)]
		sc_register_time(t, 'clust_per_t')

		# compute list of consecutive empty sal maps (cesms)
		t = cv2.getTickCount()
		cesms = []
		started = False
		current_empty_segm = []
		for i in range(d):
			if max_cl not in per_t_cls[i]:
				current_empty_segm.append(i)
				started = True
			if max_cl in per_t_cls[i]:
				if started:
					cesms.append(current_empty_segm)
					current_empty_segm = []
					started=False
		if len(current_empty_segm)>0:
			cesms.append(current_empty_segm)
		sc_register_time(t, 'clust_cesm')
		
		
		# handle empty sal maps
		t = cv2.getTickCount()
		select_cl = [max_cl] * d # init selected c. to max weight c.
		for i in range(len(cesms)):
			
			# compute 
			last_true_ind = VD['true_inds'][fc_offset+cesms[i][-1]]
			first_true_ind = VD['true_inds'][fc_offset+cesms[i][0]]
			true_fc_len = last_true_ind - first_true_ind 
							
			if true_fc_len<min_fc_cut:
				pass # will be hanlded by main method
			else:
				# create a list with 
				# the first of next clusters for each sal map
				# (cluster in per_t_cls are sorted)
				cls = []
				for j in cesms[i]:
					if len(per_t_cls[j])>0:
						cls.append(per_t_cls[j][0])
						
				# if no alternative clusters were found
				# for this segment, continue
				# this will be hanlded by main method
				if len(cls)==0:
					continue
						
				# find the most common cluster
				# it's OK if the cluster is not present in all frames
				# this will be hanlded by main method
				most_common_cl = max(cls, key=cls.count)
				
				# assign the segment's maps the new cluster and 
				# insert a new cut
				extra_cuts.append(cesms[i][0])
				for j in cesms[i]:
					select_cl[j] = most_common_cl
		sc_register_time(t, 'clust_empty_handle')
			
		# filter
		t = cv2.getTickCount()
		sal_3d_resized_backup = np.copy(sal_3d_resized)
		selX = []
		for i in range(len(X)):
			if labels[i] != select_cl[X[i][2]]:
				sal_3d_resized[X[i][0], X[i][1], X[i][2]] = 0
			else:
				selX.append([X[i][0], X[i][1], X[i][2]])
		selX = np.array(selX)
		sc_register_time(t, 'clust_filtering')

		# morphological close
		t = cv2.getTickCount()
		closing_kernel = np.ones((5, 5), np.uint8)
		if close:
			for i in range(d):
				sal_3d_resized[:,:,i] = cv2.morphologyEx(np.squeeze(sal_3d_resized[:,:,i]), 
														cv2.MORPH_CLOSE,
														closing_kernel)
		sc_register_time(t, 'clust_closing')

		# check for empty sal maps
		if verbose:
			print('   - segment (selected cluster %d)' % (max_cl))
			for i in range(d):
				empty_str=' '
				if np.sum(sal_3d_resized[:,:,i])==0:
					empty_str='*'
				print('     %3d %s %-16s (selected=%d)' % (i,empty_str, str(per_t_cls[i]), select_cl[i]))
		
	# plot after	
	t = cv2.getTickCount()	
	if plots_fn:
		ax2 = fig.add_subplot(122, projection='3d')
		sh = ax2.scatter(selX[:, 1], selX[:, 2], selX[:, 0], marker="s", s=40, cmap="RdBu")
		ax2.set_xlabel('x')
		ax2.set_ylabel('t')
		ax2.set_zlabel('y')
		ax2.set_xlim((0, sal_3d_resized.shape[1]))
		ax2.set_ylim((0, sal_3d_resized.shape[2]))
		ax2.set_zlim((0, sal_3d_resized.shape[0]))
		ax2.grid(True)
	sc_register_time(t, 'clust_plot')

	# save figure and clean up
	t = cv2.getTickCount()
	if plots_fn:
		plt.savefig(plots_fn,bbox_inches='tight')
		plt.close(fig)
		del fig
	sc_register_time(t, 'clust_plot')
	
	# resize back
	if factor==1.0:
		sc_register_time(cv2.getTickCount(), 'clust_resize_2')
		return sal_3d_resized, extra_cuts
	t = cv2.getTickCount()
	sal_3d_map.fill(0)
	for i in range(d):
		sal_3d_map[:, :, i] = cv2.resize(sal_3d_resized[:, :, i], (ow, oh), interpolation=cv2.INTER_LINEAR)
	sc_register_time(t, 'clust_resize_2')
	
	return sal_3d_map, extra_cuts

def sc_clustering_filt(hdbs_clusterer, sal_map, CP,	verbose=False):
	# if an empty saliency volume was given, return it
	if np.sum(sal_map)==0:
		return sal_map
		
	# parameters aliases for quick reference			
	factor=CP['resize_factor']
	select_sum=CP['select_sum']
	bias=CP['value_bias']
	close=CP['op_close']
								
	# resize for faster operation
	t = cv2.getTickCount()
	initH = sal_map.shape[0]
	initW = sal_map.shape[1]
	if factor!=1.0:
		sal_map = cv2.resize(sal_map, None, fx=1.0/factor, fy=1.0/factor, interpolation=cv2.INTER_LINEAR)
	sc_register_time(t, 'clust_resize_1')

	# init & gather points
	t = cv2.getTickCount()
	coo = coo_matrix(sal_map).tocoo()
	X = np.vstack((coo.row, coo.col, coo.data)).transpose()
	W = coo.data.transpose()
	sc_register_time(t, 'clust_gather')
	
	# check dimensions of X
	if len(X.shape)<2:
		print(sal_map)
		print(X)
		print(X.shape)
		input('...')
	
	# bias value of saliency map
	t = cv2.getTickCount()
	max_dim = max([initH / factor, initW / factor])
	X[:, 2] = ((X[:, 2] / np.amax(X[:, 2])) * max_dim * bias).astype(np.uint8)
	sc_register_time(t, 'clust_bias')


	if X.shape[0] > CP['hdbscan_min'] + 1:
		# cluster
		t = cv2.getTickCount()
		labels = hdbs_clusterer.fit_predict(X)
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		sc_register_time(t, 'clust_hdbscan')
		
		if n_clusters > 0:
		
			# calculate weight of each cluster
			t = cv2.getTickCount()
			weights = [0] * n_clusters
			for i in range(n_clusters):
				inds = list(np.where(labels == i)[0])
				if select_sum == 1:
					weights[i] = np.sum(W[inds])
				else:
					weights[i] = np.amax(W[inds])
			max_cl = weights.index(max(weights))
			sc_register_time(t, 'clust_weighting')

			# filter
			t = cv2.getTickCount()
			for i in range(len(X)):
				if labels[i] != max_cl:
					sal_map[X[i][0], X[i][1]] = 0
			sc_register_time(t, 'clust_filter')

			# morphological close
			t = cv2.getTickCount()
			if close:
				closing_kernel = np.ones((5, 5), np.uint8)
				sal_map = cv2.morphologyEx(sal_map, cv2.MORPH_CLOSE, closing_kernel)
			sc_register_time(t, 'clust_closing')

	# scale back image
	if factor==1.0:
		sc_register_time(cv2.getTickCount(), 'clust_resize_2')
		return sal_map
	t = cv2.getTickCount()
	sal_map = cv2.resize(sal_map, (initW, initH), interpolation=cv2.INTER_LINEAR)
	sc_register_time(t, 'clust_resize_2')
	
	return sal_map
	
def sc_clustering_filt_fast(sal_map, CP, verbose=False):
	# if an empty saliency volume was given, return it
	if np.sum(sal_map)==0:
		return sal_map
		
	# parameters aliases for quick reference			
	factor=CP['resize_factor']
	select_sum=CP['select_sum']
	bias=CP['value_bias']
	close=CP['op_close']
								
	# resize for faster operation
	t = cv2.getTickCount()
	initH = sal_map.shape[0]
	initW = sal_map.shape[1]
	if factor!=1.0:
		sal_map = cv2.resize(sal_map, None, fx=1.0/factor, fy=1.0/factor, interpolation=cv2.INTER_LINEAR)
	sc_register_time(t, 'clust_resize_1')

	# init & gather points
	t = cv2.getTickCount()
	coo = coo_matrix(sal_map).tocoo()
	X = np.vstack((coo.row, coo.col, coo.data)).transpose()
	W = coo.data.transpose()
	sc_register_time(t, 'clust_gather')
	
	# check dimensions of X
	if len(X.shape)<2:
		print(sal_map)
		print(X)
		print(X.shape)
		input('...')
	
	# bias value of saliency map
	t = cv2.getTickCount()
	max_dim = max([initH / factor, initW / factor])
	X[:, 2] = ((X[:, 2] / np.amax(X[:, 2])) * max_dim * bias).astype(np.uint8)
	sc_register_time(t, 'clust_bias')

	# clustering: call multi-pytorch-kmeans	
	labels,centers,distances,n_clusters=sc_multi_kmeans(X,verbose=verbose)
		
	if n_clusters > 0:
		
		# calculate weight of each cluster
		t = cv2.getTickCount()
		weights = [0] * n_clusters
		for i in range(n_clusters):
			inds = list(np.where(labels == i)[0])
			if select_sum == 1:
				weights[i] = np.sum(W[inds])
			else:
				weights[i] = np.amax(W[inds])
		max_cl = weights.index(max(weights))
		sc_register_time(t, 'clust_weighting')

		# filter
		t = cv2.getTickCount()
		for i in range(len(X)):
			if labels[i] != max_cl:
				sal_map[X[i][0], X[i][1]] = 0
		sc_register_time(t, 'clust_filter')

		# morphological close
		t = cv2.getTickCount()
		if close:
			closing_kernel = np.ones((5, 5), np.uint8)
			sal_map = cv2.morphologyEx(sal_map, cv2.MORPH_CLOSE, closing_kernel)
		sc_register_time(t, 'clust_closing')

	# scale back image
	if factor==1.0:
		sc_register_time(cv2.getTickCount(), 'clust_resize_2')
		return sal_map
	t = cv2.getTickCount()
	sal_map = cv2.resize(sal_map, (initW, initH), interpolation=cv2.INTER_LINEAR)
	sc_register_time(t, 'clust_resize_2')
	
	return sal_map

def sc_find_center_of_mass(sal_map, km=True, factor=2.0, bias=1.0, verbose=False):
	# if not kmeans is selected just return position of max val
	if not km:
		t = cv2.getTickCount()
		max_val = np.amax(sal_map)
		if max_val>0:
			[y, x] = np.unravel_index(sal_map.argmax(), sal_map.shape)
		else:
			x = None
			y = None
		sc_register_time(t, 'center_of_mass_gather')
		t = cv2.getTickCount()
		sc_register_time(t, 'center_of_mass_km')
		t = cv2.getTickCount()
		sc_register_time(t, 'center_of_mass_resize')
		return x, y
			
	# resize for faster operation
	t = cv2.getTickCount()
	initH = sal_map.shape[0]
	initW = sal_map.shape[1]
	sal_map = cv2.resize(sal_map, None, fx=1/factor, fy=1/factor, interpolation=cv2.INTER_NEAREST)
	sc_register_time(t, 'center_of_mass_resize')
	
	# save time for gather
	t = cv2.getTickCount()
	
	# find max val and its indicies as the initial cluster center
	max_val = np.amax(sal_map)
	[max_row, max_col] = np.unravel_index(sal_map.argmax(), sal_map.shape)
		
	# init & gather points
	coo = coo_matrix(sal_map).tocoo()
	X = np.vstack((coo.row, coo.col, coo.data)).transpose().astype(float)
	max_dim = max([initH / factor, initW / factor])
	
	# register time for gather
	sc_register_time(t, 'center_of_mass_gather')

	# cluster
	if X.shape[0] > 0:
		t = cv2.getTickCount()
		X[:, 2] = (X[:, 2] / np.amax(X[:, 2])) * max_dim * bias
		X = X.astype(np.uint8)
		clusterer = KMeans(n_clusters=1, random_state=0,
						   init=np.array([[max_row, max_col, max_val]]),
						   n_init=1,
						   max_iter=5).fit(X)
		# scale back
		x = clusterer.cluster_centers_[0][1]*factor
		y = clusterer.cluster_centers_[0][0]*factor
		sc_register_time(t, 'center_of_mass_km')
	else:
		return None, None

	# return scaled back
	return x,y

def sc_handle_empty_centers(VD, verbose=False):
	# compute list of consecutive empty centers (ecs)
	ecs = []
	started = False
	current_empty_segm = []
	for i in range(VD['fc_sel']):
		if VD['dx'][i] is None:
			current_empty_segm.append(i)
			started = True
		if VD['dx'][i] is not None:
			if started:
				ecs.append(current_empty_segm)
				current_empty_segm = []
				started=False
	if len(current_empty_segm)>0:
		ecs.append(current_empty_segm)

		
	# continue only if ecs found		
	if len(ecs)>0:
		# get starts and ends of segmentation
		starts = []
		ends = []
		for s in VD['segmentation_sel']:
			starts.append(s[0])
			ends.append(s[1])
			
		# handle each consecutive empty centers
		for i in range(len(ecs)):
			
			# print info
			if verbose:
				print('   - segment %d' % i, ecs[i], end='')
			
			# get closer distance to a segmentation start
			min_ind = min(ecs[i])
			closer_dist_2_start = min([abs(x-min_ind) for x in starts])
			
			# get closer distance to a segmentation end
			max_ind = max(ecs[i])
			closer_dist_2_end = min([abs(x-max_ind) for x in ends])
			
			# print info
			if verbose:
				print(' (min start d:%d, min end d:%d)' % \
							(closer_dist_2_start,closer_dist_2_end),
							end='')	
						
			# if the empty segment is closer to the start of a segment
			# then fill next values, else fill with previous values
			if closer_dist_2_start<closer_dist_2_end:
				x_fill_value = VD['dx'][max_ind+1] 
				y_fill_value = VD['dy'][max_ind+1] 
				if verbose:
					print(' filling next (at %d)' % (max_ind+1))
			else:
				x_fill_value = VD['dx'][min_ind-1] 
				y_fill_value = VD['dy'][min_ind-1] 
				if verbose:
					print(' filling previous (at %d)' % (min_ind-1))
				
			# fill values
			for j in range(len(ecs[i])):
				 VD['dx'][ecs[i][j]] = x_fill_value
				 VD['dy'][ecs[i][j]] = y_fill_value
				 
	# sanity check
	if verbose:
		found_errors = False
		for i in range(VD['fc_sel']):
			if VD['dx'][i] is None:
				found_errors = True
				print('!!! ', VD['dx'],'at', i)
			if VD['dy'][i] is None:
				found_errors = True
				print('!!! ', VD['dy'],'at', i)
		if found_errors:
			input('...')

	return VD

# Methods for computing mean saliency and coverage score
# used for early stopping and resorting to padded version of video
def sc_compute_mean_sal(vid_data, crop_params):
	# check mean saliency to decide whether to continue
	vid_data['mean_sal_score'] = np.average(vid_data['smaps'])
	return vid_data

def sc_compute_cvrg_score(vid_data, crop_params):
	cvrg_scores = []
	for iii in range(vid_data['fc_sel']):
		if conversion_mode == 1:
			total_sal_flat = np.sum(vid_data['smaps'][:,:,iii], 
								axis=0).reshape(1, vid_data['w_process'])
		else:
			total_sal_flat = np.sum(vid_data['smaps'][:,:,iii], 
								axis=1).reshape(1, vid_data['h_process'])
		t_sum = np.sum(total_sal_flat)
		max_cvrg = 0.0
		for d in range(total_sal_flat.shape[1] - vid_data['dim_process']):
			b_sum = np.sum(total_sal_flat[0, d:(d + vid_data['dim_process'])])
			current_cvrg = b_sum / t_sum
			if current_cvrg > max_cvrg:
				max_cvrg = current_cvrg
		cvrg_scores.append(max_cvrg)
	vid_data['mean_cvrg_score'] = sum(cvrg_scores) / len(cvrg_scores)
	
	return vid_data



# Methods for checking if extra cuts are to be inserted 
# based on the jump of focus center on the saliency map
def get_points_on_line(p1x, p1y, p2x, p2y, imageW, imageH, min_d=10):
	# difference and absolute difference between points
	# used to calculate slope and relative location between points
	dX = p2x - p1x
	dY = p2y - p1y
	dXa = np.abs(dX)
	dYa = np.abs(dY)
	
	# distance is smaller than a few pixels return None
	if dXa<min_d and dYa<min_d:
		return None

	# predefine numpy array for output based on distance between points
	itbuffer = np.empty(shape=(int(math.ceil(np.maximum(dYa,dXa))),2),dtype=np.float32)
	itbuffer.fill(np.nan)

	# Obtain coordinates along the line using a form of Bresenham's algorithm
	negY = p1y>p2y
	negX = p1x>p2x
	if p1x == p2x: #vertical line segment
		itbuffer[:,0] = p1x
		if negY:
			itbuffer[:,1] = np.arange(p1y-1, p1y-dYa-1, -1)
		else:
			itbuffer[:,1] = np.arange(p1y+1, p1y+dYa+1)				  
	elif p1y == p2y: #horizontal line segment
		itbuffer[:,1] = p1y
		if negX:
			itbuffer[:,0] = np.arange(p1x-1, p1x-dXa-1, -1)
		else:
			itbuffer[:,0] = np.arange(p1x+1, p1x+dXa+1)
	else: #diagonal line segment
		steepSlope = dYa > dXa
		if steepSlope:
			slope = dX.astype(np.float32)/dY.astype(np.float32)
			if negY:
				itbuffer[:,1] = np.arange(p1y-1, p1y-dYa-1, -1)
			else:
				itbuffer[:,1] = np.arange(p1y+1, p1y+dYa+1)
			itbuffer[:,0] = (slope*(itbuffer[:,1]-p1y)).astype(np.int) + p1x
		else:
			slope = dY.astype(np.float32)/dX.astype(np.float32)
			if negX:
				itbuffer[:,0] = np.arange(p1x-1, p1x-dXa-1, -1)
			else:
				itbuffer[:,0] = np.arange(p1x+1, p1x+dXa+1)
			itbuffer[:,1] = (slope*(itbuffer[:,0]-p1x)).astype(np.int) + p1y

	#Remove points outside of image
	colX = itbuffer[:,0]
	colY = itbuffer[:,1]
	itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]
	return itbuffer
	
def sc_check_for_extra_cuts(CP, sal_img, prevX, prevY, curX, curY, 
		verbose=False):

	# print info 
	if verbose:
		print('   jump (%3d,%3d) -> (%3d,%3d)' % (prevX, prevY, curX, curY), end='')
	# get points on line connecting two centers
	h = sal_img.shape[0]
	w = sal_img.shape[1]
	points = get_points_on_line(prevX, prevY, curX, curY, w, h, min_d=CP['min_d_jump'])
	
	# bail out if very few points on line
	if points is None:
		if verbose:
			print(' ...skipping small jump')
		return 255
	
	# get saliency sum of points under forcus movement lines
	sal_pix_sum = 0.0
	no_of_points = 0
	for i in range(points.shape[0]):
		if np.isnan(points[i,0]) or points[i,0] is None:
			continue
		no_of_points += 1
		ii = math.floor(points[i,0])
		jj = math.floor(points[i,1])
		#print('(%d,%d)=%.3f,'%(ii,jj,sal_img[jj,ii]),end='')
		sal_pix_sum += sal_img[jj,ii]

	if (no_of_points>0):
		mean_sal_jump = float(sal_pix_sum) / float(no_of_points)
	else:
		mean_sal_jump = 255
	
	if verbose:
		print('    c:%d, sum:%4.1f, mean:%4.1f' % (no_of_points, 
												sal_pix_sum, 
												mean_sal_jump))
				
		if False:		
			img = cv2.cvtColor(sal_img, cv2.COLOR_GRAY2BGR)
			for i in range(no_of_points):
				ii = math.floor(points[i,0])
				jj = math.floor(points[i,1])
				img[jj,ii,1]=255
		
			cv2.circle(img, (int(prevX),int(prevY)), 3, (0,255,0), thickness=1)
			cv2.circle(img, (int(curX),int(curY)), 5, (0,255,0), thickness=-1)
			img = cv2.resize(img, (0,0), fx=2.0, fy=2.0)
			
			DEMO_FONT = cv2.FONT_HERSHEY_SIMPLEX
			DEMO_FONT_SCALE = 0.4
			DEMO_FONT_COLOR = (255, 255, 255)
			DEMO_FONT_POS = (2, 15)
			DEMO_LINE_COLOR = (0, 255, 0)

			cv2.imshow('saliency jump', img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
	
	return mean_sal_jump
	
def sc_insert_cuts(VD, extra_cuts_at, extra_cuts_scores, 
		no_extra_cuts=10, verbose=False):
			
	# print info
	if verbose:
		print('   current segmentations::')
		print('   segmenatation:')
		print(VD['segmentation'])
		print('   segmentation_sel:')
		print(VD['segmentation_sel'])
		
	# check if we must limit the cuts
	if no_extra_cuts is not None:
		
		# sort cuts based on their weight
		extra_cuts_at = [x for _,x in sorted(zip(extra_cuts_scores,extra_cuts_at))]
		extra_cuts_scores.sort()
		
		# keep only "no_extra_cuts" cuts
		del extra_cuts_at[:no_extra_cuts]
		del extra_cuts_scores[:no_extra_cuts]
	
	# get old cuts in segmentation_sel
	old_cuts = []
	for i in range(len(VD['segmentation_sel'])):
		old_cuts.append(VD['segmentation_sel'][i][0])
		
	# append new cuts
	cuts = old_cuts + extra_cuts_at
	
	# select unique cuts
	cuts = list(set(cuts))
	
	# sort cuts
	cuts.sort()
	if verbose:
		print('   cuts:', cuts)
	
	# re-compute segmentation_sel
	old_end = VD['segmentation_sel'][-1][1]
	VD['segmentation_sel'] = []
	for i in range(len(cuts)-1):
		VD['segmentation_sel'].append([cuts[i], cuts[i+1]-1])
	VD['segmentation_sel'].append([cuts[-1], old_end])
	VD['segmentation_sel'] = np.array(VD['segmentation_sel'])

	# map cuts to original frame indices
	cuts = [VD['true_inds'][x] for x in cuts]
	
	# re-compute segmentation
	old_end = VD['segmentation'][-1][1]
	VD['segmentation'] = []
	for i in range(len(cuts)-1):
		VD['segmentation'].append([cuts[i], cuts[i+1]-1])
	VD['segmentation'].append([cuts[-1], old_end])
	VD['segmentation'] = np.array(VD['segmentation'])

	# print info
	if verbose:
		print('   re-computed segmentations::')
		print('   segmenatation:')
		print(VD['segmentation'])
		print('   segmentation_sel:')
		print(VD['segmentation_sel'])
	
	return VD, extra_cuts_at, extra_cuts_scores



# Methods for interpolationg, low-passing 
# and smoothing displacement arrays
def interp_handler(d, sampled_t, true_t):
	l=len(d)
	
	# if segment length is one, perform repeat interpolation
	if l<3:
		di = [float(d[0])] * len(true_t)
		return di
		
	# if segment is small, perform linear interpolation
	if l>=3 and l<=6:
		f = interpolate.interp1d(sampled_t, d, 
				fill_value="extrapolate",kind="linear")
		di = list(f(true_t))	
		return di
	
	# quadratic interpolation 
	f = interpolate.interp1d(sampled_t, d, 
				fill_value="extrapolate",kind="quadratic")
	di = list(f(true_t))	
	return di
			
def sc_interpolate(vid_data, crop_params, verbose=False):
	# init interpolated segments and total series
	vid_data['dxi'] = []
	vid_data['dyi'] = []

	# interpolate each segment separately
	l = vid_data['segmentation_sel'].shape[0]
	for i in range(l):
		
		# gather sampled and true timestamps based on segmentation
		si = vid_data['segmentation'][i][0]
		ei = vid_data['segmentation'][i][1]+1
		sis = vid_data['segmentation_sel'][i][0]
		eis = vid_data['segmentation_sel'][i][1]+1
		sampled_t = vid_data['true_inds'][sis:eis]
		true_t = np.arange(0,ei-si)
		
		# make time variable zero based
		min_ind = min(sampled_t)
		sampled_t = [x-min_ind for x in sampled_t]
		
		# compute segment length
		cl = ei-si
		
		# gather x and offsets on sampled segmentation
		dx = vid_data['dx'][sis:eis]
		dy = vid_data['dy'][sis:eis]
		
		# interpolate x offsets 		
		dxi = interp_handler(dx, sampled_t, true_t)

		# interpolate y offsets - on failure register mean
		dyi = interp_handler(dy, sampled_t, true_t)

		# print info
		if verbose:
			print('   - segment %d : %d -> %d (%d)' % (i,si,ei,cl))
			print('     sampled_t len = %4d,'%len(sampled_t),
						'dx len  = %4d,'%len(dx),
						'dy len  = %4d,' % len(dy))
			print('     true_t len    = %4d,'%len(true_t),
						'dxi len = %4d,' % len(dxi),
						'dyi len = %4d,' % len(dyi))
			
		# append data
		vid_data['dxi'] = vid_data['dxi'] + dxi
		vid_data['dyi'] = vid_data['dyi'] + dyi

	return vid_data

def sc_butter_lowpass_filter(x, cutoff, fs, order):
	try:
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
		try:
			return signal.filtfilt(b, a, x)
		except:
			pass
	except:
		pass

	try:
		y = np.convolve(x, np.ones(5), 'same') / 5
		for i in range(2, len(x) - 2):
			x[i] = y[i]
		return x
	except:
		pass
		
	try:
		y = np.convolve(x, np.ones(3), 'same') / 5
		for i in range(2, len(x) - 2):
			x[i] = y[i]
		return x
	except:
		pass
		
	return x

def loess_handler(t_vec, di, loess_filt, adj_window, degree):
	cl = len(t_vec)
	
	if cl<10:
		return list(di)
	
	if loess_filt:
		cl = len(t_vec)
		loess = pyloess.Loess(t_vec, di)
		ds = [loess.estimate(j, window=adj_window, use_matrix=False, degree=degree) for j in range(cl)]
		if np.isnan(np.sum(ds)):
			ds = list(di)
		return ds

	return list(di)
							
def sc_smoothing(vid_data, loess_filt, window_to_fr, degree, 
				lp_filt, lp_cutoff, lp_order, verbose=False):
	vid_data['dxs'] = []
	vid_data['dys'] = []
	sr = vid_data['fr']
	l = vid_data['segmentation_sel'].shape[0]
	
	# smooth each segment separately
	for i in range(l):

		# gather sampled and true timestamps based on full segmentation
		si = vid_data['segmentation'][i][0]
		ei = vid_data['segmentation'][i][1]+1
		cl = ei-si
		t_vec = np.array(list(range(cl)))

		# adjust window based on segment length
		adj_window = min(int(vid_data['fr']*window_to_fr), cl-2)
		if (adj_window % 2) == 0:
			adj_window -= 1

		# print info
		if verbose:
			print('   - segment %d : %d -> %d (len=%d, window=%d)' % (i,si,ei,cl,adj_window))
			
		# get x and pad end with last value if this is the last segment
		dxi = np.array(vid_data['dxi'][si:ei])
		if len(dxi)<cl and i==l-1:
			old_l = len(dxi)
			last_val = dxi[-1]
			dxi = np.resize(dxi,cl)
			for j in range(old_l,cl):
				dxi[i]=last_val
				
		# low pass x	
		dxi_lp = sc_butter_lowpass_filter(dxi, lp_cutoff, sr, lp_order) \
									if lp_filt else dxi


		# loess on x
		dxs = loess_handler(t_vec, dxi_lp, loess_filt, adj_window, degree)

				

					
		# get x and pad end with last value if this is the last segment
		dyi = np.array(vid_data['dyi'][si:ei])
		if len(dyi)<cl and i==l-1:
			old_l = len(dyi)
			last_val = dyi[-1]
			dyi = np.resize(dyi,cl)
			for j in range(old_l,cl):
				dyi[i]=last_val
				
		# low pass y
		dyi_lp = sc_butter_lowpass_filter(dyi, lp_cutoff, sr, lp_order) \
									if lp_filt else dyi


		# loess on y
		dys = loess_handler(t_vec, dyi_lp, loess_filt, adj_window, degree)
				
		# print info
		if verbose:
			print('     t_vec len = %4d,'%len(t_vec),
						'dxi len = %4d,'%len(dxi),
						'dxi_lp len = %4d,'%len(dxi_lp),
						'xs len = %4d'%len(dxs))
			print('     t_vec len = %4d,'%len(t_vec),
						'dyi len = %4d,'%len(dyi),
						'dyi_lp len = %4d,'%len(dyi_lp),
						'dys len = %4d'%len(dys))
			
			
		# register smoothed series
		vid_data['dxs'] = vid_data['dxs'] + dxs
		vid_data['dys'] = vid_data['dys'] + dys
		
	return vid_data



# Shifts displacements arrays in time
# to catch quick movements
def sc_shift_time(vid_data, shift):
	if shift > 0:
		for i in range(shift):
			vid_data['bbs'][-i+1] = vid_data['bbs'][-1]
		for i in range(len(vid_data['bbs']) - shift):
			vid_data['bbs'][i] = vid_data['bbs'][i + shift]
	return vid_data



# Plots infered and smoothed displacemnt arrays
# using matplotlib
def sc_plot_signals(vid_data, plots_fn):
	if not plots_fn:
		return
		
	# set figure handle
	fig = plt.figure()
	
	# get interpolated data
	dxi = vid_data['dxi']
	dyi = vid_data['dyi']
	
	# get smoothed (final) data
	dxs = vid_data['dxs']
	dys = vid_data['dys']
	
	# get shot boundaries from segementation
	shot_boundaries = [0] * len(dxi)
	for i in vid_data['segmentation']:
		shot_boundaries[i[0]]=1
	shot_boundaries[-1]=1
	
	# make 4 subplots
	for i,d in enumerate([dxi,dyi,dxs,dys]):
		# create time array
		ats = list(range(len(d)))
		
		# get shot lines array
		shot_lines = [x * max(d) for x in shot_boundaries]
		
		# plot signals
		axs = fig.add_subplot(2,2,i+1)
		pld, = axs.plot(ats, d, color=(0, 0.5, 0.7))
		axs.plot(ats, shot_lines, color=(0, 0, 0))
		axs.set_xlim(-1, len(d))
		axs.set_ylim(1, max(d))

		# smallen text for al objects on figure
		for item in ([axs.title, axs.xaxis.label, axs.yaxis.label] +
					 axs.get_xticklabels() + axs.get_yticklabels()):
			item.set_fontsize(6)
		
	# save figure and clean up
	plt.savefig(plots_fn, bbox_inches='tight')
	plt.close(fig)
	del fig


				
# Renders smart-cropped version of video and demo of the procedure
def sc_renderer(vid_data, crop_params, 
				vid_path, out_path, demo_path, 
				verbose=False):
	# aliases for quick reference
	fr 	 		= vid_data['fr']
	fc 	 		= vid_data['fc']
	frame_w		= vid_data['w_orig'] 
	frame_h 	= vid_data['h_orig'] 
	process_w	= vid_data['w_process']
	process_h   = vid_data['h_process'] 
	final_w 	= vid_data['w_final']
	final_h		= vid_data['h_final']
	bbs 		= vid_data['bbs'] # bounding in boxes are in [x1,y1,x2,y2]
	scale_h 	= float(process_h) / float(frame_h)
	scale_w 	= float(process_w) / float(frame_w)
	fbb_h       = vid_data['fbb_h']
	fbb_w       = vid_data['fbb_w']
	m           = vid_data['inds_to_orig'] # get index sal map index to original frame index
		
	# setup output video writer
	if out_path:
		out_path += '.mp4'
		final_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		out_final = cv2.VideoWriter(out_path, 
									final_fourcc, 
									fr, 
									(fbb_w, fbb_h))
		print(' output video to: %s' % (out_path))
	else:
		out_final = None

	# setup demo video writer
	if demo_path:
		demo_path += '.mp4'
		demo_fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
		out_demo = cv2.VideoWriter(demo_path, 
									demo_fourcc, 
									fr, 
									(5*process_w, process_h))
		print(' demo video to: %s' % (demo_path))
	else:
		out_demo = None
		
	# demo params
	DEMO_FONT = cv2.FONT_HERSHEY_SIMPLEX
	DEMO_FONT_SCALE = 0.4
	DEMO_FONT_COLOR = (255, 255, 255)
	DEMO_BACK_COLOR = (0, 0, 0)
	DEMO_FONT_COLOR_A = (255, 255, 255, 255)
	DEMO_BACK_COLOR_A = (0, 0, 0, 255)
	DEMO_FONT_POS = (2, 15)
	DEMO_FONT_POS2 = (2, 30)
	DEMO_FONT_POS3 = (2, 45)
	DEMO_FONT_POS4 = (2, 60)
	DEMO_FONT_POS5 = (2, 75)
	DEMO_LINE_COLOR = (0, 255, 0)
	fade_len = fr
	past_steps = int(fr/3.0)
	
	# cut faders
	fade_out_cut = 0
	fade_out_clust = 0
	fade_out_extra = 0

	# read video
	if out_path or demo_path:
		fvs = FileVideoStream(vid_path).start()
		iii = -1

		while fvs.more():
			
			# read frame
			frame = fvs.read()
			if frame is None:
				break
			iii += 1
			print('\r input read %d/%d...' % (iii+1,fc), end='')
			
			# if no available bounding boxes then stop
			if iii+1==len(bbs):
				break
				
			# get bounding box coordinates
			bx1, by1, bx2, by2 = bbs[iii]
				
			# if smart cropped video is set...
			if out_final is not None:
							
				# write output video frames
				out_final_contiguous_frame = np.ascontiguousarray(\
									np.copy(frame[by1:by2, bx1:bx2, :]))
				out_final.write(out_final_contiguous_frame)


			# if demo video is set...
			if out_demo is not None:
					
				# init demo frame sections
				framex = np.ascontiguousarray(cv2.resize(np.copy(frame), 
									(process_w, process_h)))
				frame_sal_unprocx = np.ascontiguousarray(cv2.cvtColor(\
									vid_data['smaps_orig'][:,:,m[iii]], 
									cv2.COLOR_GRAY2BGR))
				frame_sal_procedx = np.ascontiguousarray(cv2.cvtColor(\
									vid_data['smaps'][:,:,m[iii]], 
									cv2.COLOR_GRAY2BGRA))
				frame_overlayedx = np.ascontiguousarray(\
									np.copy(framex))
				frame_final_bbx = np.ascontiguousarray(\
									np.copy(framex))
									
				# overlay processed saliency map on original frame
				cv2.addWeighted(frame_overlayedx, 0.5, 
									cv2.cvtColor(frame_sal_procedx,
									cv2.COLOR_BGRA2BGR), 0.5, 
									0, frame_overlayedx)

				
				# plot current center and 5 old centers on filtered saliency
				# frame_sal_procedx
				cv2.circle(frame_sal_procedx, 
									(int(vid_data['dx'][m[iii]]),
									int(vid_data['dy'][m[iii]])), 
									5, (0,255,0,255),
									lineType=cv2.LINE_AA,
									thickness=-1)
				for j in range(1,past_steps):
					if iii-j>=0:
						current_color = (0,255-(j*10)-20,0,255-(j*30))
						cv2.circle(frame_sal_procedx,
									(int(vid_data['dx'][m[iii]-j]),
									int(vid_data['dy'][m[iii]-j])), 
									2, current_color,
									lineType=cv2.LINE_AA,
									thickness=past_steps-j)
				for j in range(1,past_steps):
					if iii-j>0:
						current_color = (0,255-(j*10)-20,0,255-(j*30))
						cv2.line(frame_sal_procedx,
									(int(vid_data['dx'][m[iii]-j]),
									int(vid_data['dy'][m[iii]-j])), 
									(int(vid_data['dx'][m[iii]-j-1]),
									int(vid_data['dy'][m[iii]-j]-1)), 
									current_color,
									lineType=cv2.LINE_AA,
									thickness=past_steps-j)
				
				# print jump score in second line
				cv2.putText(frame_sal_procedx, 'jump:%d' % \
									(vid_data['jumps'][m[iii]]), 
									DEMO_FONT_POS2, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_BACK_COLOR_A,
									lineType=cv2.LINE_AA, thickness=2)
				cv2.putText(frame_sal_procedx, 'jump:%d' % \
									(vid_data['jumps'][m[iii]]), 
									DEMO_FONT_POS2, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_FONT_COLOR_A,
									lineType=cv2.LINE_AA, thickness=1)

								
				# print original cut in third line
				if iii in vid_data['segm_backup']:
					fade_out_cut = fade_len
				if fade_out_cut>0:
					cv2.putText(frame_sal_procedx, 'orig. cut', 
									DEMO_FONT_POS3, DEMO_FONT, 
									DEMO_FONT_SCALE, 
									(0,0,0,fade_out_cut*int(255/fade_len)),
									lineType=cv2.LINE_AA, thickness=2)
					cv2.putText(frame_sal_procedx, 'orig. cut',
									DEMO_FONT_POS3, DEMO_FONT, 
									DEMO_FONT_SCALE, 
									(255,255,255,fade_out_cut*int(255/fade_len)),
									lineType=cv2.LINE_AA, thickness=1)
					fade_out_cut -= 1
									
				# print clustering cut in fourth line
				vid_data['extra_clust_cuts_at']
				if m[iii] in vid_data['extra_clust_cuts_at']:
					fade_out_clust = fade_len
				if fade_out_clust>0:
					cv2.putText(frame_sal_procedx, 'clust. cut', 
									DEMO_FONT_POS4, DEMO_FONT, 
									DEMO_FONT_SCALE,
									(0,0,0,fade_out_clust*int(255/fade_len)),
									lineType=cv2.LINE_AA, thickness=2)
					cv2.putText(frame_sal_procedx, 'clust. cut',
									DEMO_FONT_POS4, DEMO_FONT, 
									DEMO_FONT_SCALE,
									(255,255,255,fade_out_clust*int(255/fade_len)),
									lineType=cv2.LINE_AA, thickness=1)
					fade_out_clust -= 1

				# print extra cut in fifth line
				vid_data['extra_cuts_at']
				if m[iii] in vid_data['extra_cuts_at']:
					fade_out_extra = fade_len
				if fade_out_extra>0:
					cv2.putText(frame_sal_procedx, 'xtra cut', 
									DEMO_FONT_POS5, DEMO_FONT, 
									DEMO_FONT_SCALE,
									(0,0,0,fade_out_extra*int(255/fade_len)),
									lineType=cv2.LINE_AA, thickness=2)
					cv2.putText(frame_sal_procedx, 'xtra cut',
									DEMO_FONT_POS5, DEMO_FONT, 
									DEMO_FONT_SCALE, 
									(255,255,255,fade_out_extra*int(255/fade_len)),
									lineType=cv2.LINE_AA, thickness=1)
					fade_out_extra -= 1

				# now that all text is printed discard alpha channel
				# (frame_sal_procedx)
				frame_sal_procedx = cv2.cvtColor(frame_sal_procedx,
									cv2.COLOR_BGRA2BGR)

				# plot unsmoothed center on frame_overlayed
				# (dx and dy are w.r.t to process dimensions)
				dx = int(vid_data['dx'][m[iii]])
				dy = int(vid_data['dy'][m[iii]])
				frame_overlayedx = cv2.circle(frame_overlayedx, (dx,dy), 3, 
									DEMO_LINE_COLOR, -1,
									lineType=cv2.LINE_AA)
										
				# plot smoothed center on frame_final_bb
				# (final_dx and final_dy are w.r.t. to original video dimension)
				# (must rescale)
				fx = int(vid_data['dxs'][iii]*scale_w)
				fy = int(vid_data['dys'][iii]*scale_h)
				frame_final_bbx = cv2.circle(frame_final_bbx, (fx,fy), 3, 
									DEMO_LINE_COLOR, -1,
									lineType=cv2.LINE_AA)
				
				# plot final bounding box on frame_final_bb
				# (bounding boxes have the scale of the original video)
				# (must rescale)
				frame_final_bbx = cv2.rectangle(frame_final_bbx,
									(int(bx1*scale_w)+1, 
									int(by1*scale_h)+1),
									(int(bx2*scale_w)-1, 
									int(by2*scale_h)-1),
									DEMO_LINE_COLOR, 1,
									lineType=cv2.LINE_AA)
				
				# text info on frames
				cv2.putText(frame_sal_unprocx, 'saliency', 
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_BACK_COLOR, 
									lineType=cv2.LINE_AA, thickness=2)
				cv2.putText(frame_sal_unprocx, 'saliency', 
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_FONT_COLOR, 
									lineType=cv2.LINE_AA, thickness=1)
				cv2.putText(frame_sal_procedx, 'filtered',
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_BACK_COLOR, 
									lineType=cv2.LINE_AA, thickness=2)
				cv2.putText(frame_sal_procedx, 'filtered',
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_FONT_COLOR, 
									lineType=cv2.LINE_AA, thickness=1)
				cv2.putText(frame_overlayedx, 'overlayed',
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_BACK_COLOR, 
									lineType=cv2.LINE_AA, thickness=2)
				cv2.putText(frame_overlayedx, 'overlayed',
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_FONT_COLOR, 
									lineType=cv2.LINE_AA, thickness=1)
				cv2.putText(frame_final_bbx, 'final', 
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_BACK_COLOR, 
									lineType=cv2.LINE_AA, thickness=2)
				cv2.putText(frame_final_bbx, 'final', 
									DEMO_FONT_POS, DEMO_FONT, 
									DEMO_FONT_SCALE, DEMO_FONT_COLOR, 
									lineType=cv2.LINE_AA, thickness=1)
								
				# write concatenated frame
				tempx = np.concatenate((framex, frame_sal_unprocx, 
									frame_sal_procedx, frame_overlayedx, 
									frame_final_bbx), axis=1)
				out_final_contiguous_frame = np.ascontiguousarray(\
									np.copy(tempx))
				out_demo.write(out_final_contiguous_frame)
				
				# clean up
				del framex
				del frame_sal_unprocx
				del frame_sal_procedx
				del frame_overlayedx
				del frame_final_bbx
				del tempx
				del out_final_contiguous_frame
									
			# if last frame force break
			if iii+1==fc:
				break

	# free video reader
	if out_path or demo_path:
		fvs.stop()
		del fvs
		
	# free video writers
	if out_path:
		if out_final is not None:
			out_final.release()
	if demo_path:
		if out_demo is not None:
			out_demo.release()
			
	# if demo was created then enlarge and drop frame rate 
	# for more detailed view
	temp_demo_path = demo_path.replace('.mp4', '.temp.mp4')
	enlarged_h = int(process_h*1.5)
	enlarged_w = 5*int(process_w*1.5)
	if enlarged_h%2==1:
		enlarged_h+=1
	if enlarged_w%2==1:
		enlarged_w+=1
	str_ffmpeg_scale = '%dx%d' % (enlarged_h,enlarged_w)
	ffmpeg.input(demo_path) \
			.filter('fps', fps=int(fr/1.5), round='up') \
			.filter('scale', str_ffmpeg_scale) \
			.output(temp_demo_path) \
			.run(overwrite_output=True)
	os.remove(demo_path)
	os.rename(temp_demo_path, demo_path)

# Renders padded version of video
def sc_render_padded(vid_data, crop_params, vid_path, out_path, verbose=False):
	if out_path:
		return
		
	# assign video data to aliases for quick reference
	fr 	 		= vid_data['fr']
	frame_w		= vid_data['w_orig'] 
	frame_h 	= vid_data['h_orig'] 
	process_w	= vid_data['w_process']
	process_h   = vid_data['h_process'] 
	final_w 	= vid_data['w_final']
	final_h		= vid_data['h_final']

	# compute dimensions of padded video
	# (bounding boxes are in [x1,y1,x2,y2])
	c = crop_params['out_ratio'].split(':')
	height_ratio = float(c[0])
	width_ratio = float(c[1])
	if (width_ratio * frame_w) > (height_ratio * frame_h):  # pad height
		bb_w = 0
		bb_h = int(math.floor((width_ratio / height_ratio) * frame_w)) - frame_h
	else:  # pad width
		bb_w = int(math.floor((height_ratio / width_ratio) * frame_h)) - frame_w
		bb_h = 0
	half_bb_h = int(math.ceil(bb_h / 2))
	half_bb_w = int(math.ceil(bb_w/ 2))
	
	# setup output video writer
	final_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	out_final = cv2.VideoWriter(outPath, final_fourcc, fr, (bb_w, bb_h))
	print(' output video to: %s' % (out_path))
	
	# read video
	fvs = FileVideoStream(vid_path).start()
	iii = -1
	while fvs.more():
		
		# read frame
		frame = fvs.read()
		if frame is None:
			break
		iii += 1

		# pad
		padded_frame = cv2.copyMakeBorder(frame, 
										half_bb_h, half_bb_h,
										half_bb_w, half_bb_w,
										cv2.BORDER_CONSTANT, 
										value=[0, 0, 0])
									 
		# write output video frames
		out_final.write(padded_frame)
		
	# free video reader and writer
	fvs.stop()
	del fvs
	out_final.release()
	

	
	
	
	
	
def smart_vid_crop(video_path, CP=None, 
				demo_fn='', final_vid_fn='', plots_fn='', temp_path=None,
				verbose=False, save_vid=True):
	# clear registered time measurements
	sc_init_time()
	
	# init dictionary to hold results
	smart_crop_results = {}
	
	# set padding trigger
	do_pad = False
	
	# set figure filenames
	plots_clust_fn = ''
	plots_signals_fn = ''
	if plots_fn:
		plots_clust_fn = plots_fn.replace('.','_clust.')
		plots_signals_fn = plots_fn.replace('.','_signals.')

	# ensure proper crop_params
	if CP is None:
		CP = sc_init_crop_params()

	# check if features are already saved
	VD = None
	if temp_path is not None:
		VD_fn = os.path.join(temp_path, vid_fn+'.pkl')
		if os.path.isfile(VD_fn):
			print(' (loading %s)'%VD_fn)
			with open(VD_fn, 'rb') as fp:
				VD = pickle.load(fp)
			
			# get saved times and re-register
			sc_save_time_override('read_init', VD['times']['read_init'])
			sc_save_time_override('_read', VD['times']['_read'])
			sc_save_time_override('_read_shot_det', VD['times']['_read_shot_det'])
			sc_save_time_override('_read_sal_det', VD['times']['_read_sal_det'])
			sc_save_time_override('read_tidy', VD['times']['read_tidy'])

	# read video, perform shot detection and saliency detection
	if VD is None:
		VD = read_and_segment_video(video_path, CP, verbose=verbose)
		
	# check if we must save features
	if temp_path is not None:
		if not os.path.isfile(VD_fn):
			print(' (saving %s)'%VD_fn)
			with open(VD_fn, 'wb') as fp:
				pickle.dump(VD, fp)
				
	# save segmenatation backup for demo purposes
	VD['segm_backup'] = VD['segmentation'].copy()
				
	# compute final crop window
	t = cv2.getTickCount()
	print(' - Destination size calculation... ')
	VD = sc_calc_dest_size(VD, CP, verbose=verbose)
	sc_register_time(t, '_calc_dest_size')

	# check for blank borders in the video
	t = cv2.getTickCount()
	print(' - Border detection... ')
	VD = sc_border_detection(CP, VD, verbose=verbose)
	sc_register_time(t, '_border_det')
	
	# check mean saliency for resorting to padding
	t = cv2.getTickCount()
	if CP['exit_on_spread_sal']:
		print(' - Checking mean saliency... ')
		VD = sc_compute_mean_sal(VD,CP)
		if VD['mean_sal']>CP['t_sal']:
			do_pad = True
			print('   (mean saliency: %.3f > %.3f - skipping...)' % \
							(VD['mean_sal_score'], CP['t_sal']))
		else:
			print('   (mean saliency: %.3f < %.3f - continuing...)' % \
							(VD['mean_sal_score'], CP['t_sal']))
	else:
		VD['mean_sal_score'] = None
	sc_register_time(t, '_check_mean_sal')
	
	# threshold saliency map	
	t = cv2.getTickCount()
	if not do_pad:
		print(' - Thresholding...')
		# save a copy of sal. maps to "smaps_orig" 
		# for visualization purposes
		VD = sc_threshold(VD, CP, copy=not (demo_fn==''))
	sc_register_time(t, '_thresh')
	
	# clustering
	t = cv2.getTickCount()
	hdbs_clusterer = hdbscan.HDBSCAN(min_cluster_size=CP['hdbscan_min'], allow_single_cluster=True)
	sc_register_time(t, 'clust_init')

	t = cv2.getTickCount()
	total_clust_cuts = []
	if not do_pad:
		if not CP['clust_filt']:
			print(' - Skipping clustering... ')
		else:
			if CP['use_3d']:
				print(' - Clustering 3D... ')
				for i,s in enumerate(VD['segmentation_sel']):
					fc_offset = 0 if i==0 else VD['segmentation_sel'][i-1][1]+1
					if CP['use_fast']:
						VD['smaps'][:, :, s[0]:s[1]+1], segment_extra_cuts = \
							sc_clustering_filt_3d_fast(VD['smaps'][:,:,s[0]:s[1]+1],
														CP, VD, fc_offset,
														plots_fn=plots_clust_fn, 
														verbose=verbose)	
					else:
						VD['smaps'][:, :, s[0]:s[1]+1], segment_extra_cuts = \
							sc_clustering_filt_3d(hdbs_clusterer,
								VD['smaps'][:,:,s[0]:s[1]+1],
								CP, VD, fc_offset,
								plots_fn=plots_clust_fn, 
								verbose=verbose)	
							
					# add start of segment to extrac cuts indices
					# to reflect true indices
					segment_extra_cuts = [x+s[0] for x in segment_extra_cuts]
					
					# add current segment's cuts to total cuts list
					total_clust_cuts = total_clust_cuts + segment_extra_cuts
					
			else:
				print(' - Clustering... ')
				for i in range(VD['fc_sel']):
					if CP['use_fast']:
						VD['smaps'][:,:,i] = \
							sc_clustering_filt_fast(VD['smaps'][:,:,i],
													CP,
													verbose=verbose)
					else:
						VD['smaps'][:,:,i] = \
							sc_clustering_filt(hdbs_clusterer,
								VD['smaps'][:,:,i],
								CP,
								verbose=verbose)
	smart_crop_results['cuts_clust'] = len(total_clust_cuts)
	sc_register_time(t, '_clustering')
	
	# check coverage score for resorting to padding
	t = cv2.getTickCount()
	if not do_pad:
		if CP['exit_on_low_cvrg']:
			print(' - Checking coverage score... ')
			VD = sc_compute_cvrg_score(VD, CP)
			if VD['mean_cvrg_score'] < CP['t_cvrg']:
				do_pad = True
				print('   (mean cvrg: %.3f > %.3f - skipping...)' % \
							(VD['mean_cvrg_score'], CP['t_cvrg']))
			else:
				print('   (mean cvrg: %.3f < %.3f - continuing...)' % \
							(VD['mean_cvrg_score'], CP['t_cvrg']))
		else:
			VD['mean_cvrg_score'] = None
	else:
		VD['mean_cvrg_score'] = None
	sc_register_time(t, '_check_cvrg')
			
	# get center of mass for each frame
	t = cv2.getTickCount()
	if not do_pad:
		print(' - Center of mass... ')
		VD['dx'] = []
		VD['dy'] = []
		for i in range(VD['fc_sel']):
			if np.sum(VD['smaps'][:,:,i]) > 0:
				dx, dy = sc_find_center_of_mass(VD['smaps'][:,:,i],
												km=CP['com_km'],
												factor=CP['resize_factor'],
												bias=CP['value_bias'], 
												verbose=verbose)
			else:
				print('   (empty sal map at pos %d)' % i)
				dx = None
				dy = None
			VD['dx'].append(dx)
			VD['dy'].append(dy)
	sc_register_time(t, '_center_of_mass')
	
	# hanlde not found centers
	t = cv2.getTickCount()
	if not do_pad:
		print(' - Empty centers handling... ')
		VD = sc_handle_empty_centers(VD, verbose=verbose)
	sc_register_time(t, '_center_empty_handle')
	
	# check if we must insert extra cuts that result from 3d clustering
	t = cv2.getTickCount()
	VD['extra_clust_cuts_at'] = []
	if not do_pad:
		if len(total_clust_cuts)>0:
			print(' - Clustering cuts (%d)...' % (len(total_clust_cuts)))
			
			# insert all clustering cuts - do not use scores
			# (i.e. scores = [] and no_extra_cuts=None
			VD, extra_clust_cuts_at, extra_clust_cuts_scores = \
										sc_insert_cuts(VD,
											total_clust_cuts, [],
											no_extra_cuts=None,
											verbose=verbose)
										
			# register clustering extra cuts in video data
			VD['extra_clust_cuts_at'] = extra_clust_cuts_at
	sc_register_time(t, '_insert_clust_cuts')
		
	# init jumps and extra cuts arrays
	VD['jumps'] = [255] * len(VD['dx'])
	VD['extra_cuts_at'] = []
	smart_crop_results['cuts_extra'] = 0
	smart_crop_results['no_extra_cuts'] = CP['no_extra_cuts']
	
	# check if we must insert extra "focus change" cuts
	t = cv2.getTickCount()
	if not do_pad:
		if CP['no_extra_cuts']>0:
			print(' - Checking for focus change extra cuts... ')
			extra_cuts_at = []
			extra_cuts_scores = []
			
			# get cuts of segmentation
			segm_cuts = []
			for i in range(VD['segmentation_sel'].shape[0]):
				segm_cuts.append(VD['segmentation_sel'][i,0]-1)
				segm_cuts.append(VD['segmentation_sel'][i,0])
				segm_cuts.append(VD['segmentation_sel'][i,1])
				segm_cuts.append(VD['segmentation_sel'][i,1]+1)
				
			# cycle though frames (sal. maps)
			# and check center of mass jumps
			tt = cv2.getTickCount()
			if verbose:
				print('   (sal. map shape:', VD['smaps'][:,:,0].shape, ')')
			for i in range(1,VD['fc_sel']):
				mean_jump = sc_check_for_extra_cuts(CP, VD['smaps'][:,:,i],
												VD['dx'][i-1],VD['dy'][i-1], 
												VD['dx'][i],VD['dy'][i], 
												verbose=verbose)
														
				# if jump (=mean saliency of jumped pixels) is very low ...
				if mean_jump<CP['t_cut']:
					
					# and current frame is not a cut of segmentation
					if i not in segm_cuts:
					
						# then add current frame as extra cut
						extra_cuts_at.append(i)
						extra_cuts_scores.append(mean_jump)
				
				# register mean saliency jump for demo purposes
				VD['jumps'][i] = mean_jump
				
			sc_register_time(tt, 'find_extra_cuts')

			# do extra cuts in segmentation
			tt = cv2.getTickCount()
			if len(extra_cuts_at)>0:
				smart_crop_results['cuts_extra'] = len(extra_cuts_at)
				VD, extra_cuts_at, extra_cuts_scores = sc_insert_cuts(VD,
										extra_cuts_at, extra_cuts_scores,
										no_extra_cuts=CP['no_extra_cuts'],
										verbose=verbose)
										
			# register extra cuts in video data 
			# (filtered by sc_insert_cuts method)
			VD['extra_cuts_at'] = extra_cuts_at
			VD['extra_cuts_scores'] = extra_cuts_scores
			sc_register_time(tt, 'insert_extra_cuts')
	sc_register_time(t, '_extra_cuts')
		
	# interpolate
	t = cv2.getTickCount()
	if not do_pad:
		print(' - Interpolating...')
		VD = sc_interpolate(VD, CP, verbose=verbose)
	sc_register_time(t, '_interpolation')
	
	# smooth signals
	t = cv2.getTickCount()
	if not do_pad:
		print(' - Smoothing...')
		VD = sc_smoothing(VD,
				CP['loess_filt'], CP['loess_w_secs'], CP['loess_degree'],
				CP['lp_filt'], CP['lp_cutoff'], CP['lp_order'], 
				verbose=verbose)
	sc_register_time(t, '_smooth')
			
	# plot signals, infered and smoothed
	if plots_fn:
		print(' - Plotting series...')
		sc_plot_signals(VD, plots_fn=plots_signals_fn)


	# compute bounding boxes from center coordinates
	# ALL segments coordinates are now scaled back to original size
	# bounding in boxes are in [x1,y1,x2,y2]
	print(' - Computing boudning boxes...')
	t = cv2.getTickCount()
	VD = sc_compute_bb(VD, CP, verbose=verbose)
	sc_register_time(t, '_bb')


	# shift time series backwards to catch up fast movements
	t = cv2.getTickCount()
	if not do_pad:
		if CP['shift_time']>0:
			print(' - Time shift...')
			VD = sc_shift_time(VD, CP['shift_time'])
	sc_register_time(t, '_shift')


	t = cv2.getTickCount()
	if do_pad:
		if save_vid:
			print(' - Rendering padded video')
			sc_render_padded(VD, CP, 
							 vid_path, final_vid_fn,
							 verbose=verbose)
		smart_crop_results['result'] = 'padded'
	else:
		# render final video and demo
		if save_vid:
			print(' - Rendering outputs')
			sc_renderer(VD, CP,
						vid_path, final_vid_fn, demo_fn,
						verbose=verbose)
		smart_crop_results['result'] = 'smart cropped'
	sc_register_time(t, 'render')

	# JSON info
	info_string = ' (%dx%d)->(%dx%d)->(%dx%d)->(%dx%d)\n' % \
									(VD['h_orig'], VD['w_orig'], 
									 VD['h_process'], VD['w_process'], 
									 VD['h_final'], VD['w_final'], 
									 VD['fbb_h'], VD['fbb_w'])
	smart_crop_results['info'] = info_string

	# JSON parameters
	params_string = ''
	for cpk in CP.keys():
		params_string += ' %-18s : %s\n' % (cpk, str(CP[cpk]))
	smart_crop_results['params'] = params_string

	# JSON scores
	smart_crop_results['mean_sal_score'] = VD['mean_sal_score']
	smart_crop_results['mean_sal_score_t'] = CP['t_sal']
	smart_crop_results['coverage_score'] = VD['mean_cvrg_score']
	smart_crop_results['coverage_score_t'] = CP['t_cvrg']
	
	# JSON times (and print)
	print(' Times::')
	t_dict = sc_all_times(VD['fc'] / VD['fr'])
	for k in t_dict.keys():
		if k.startswith('_'):
			smart_crop_results['t_'+k] = t_dict[k]
			print('   %-21s : %s' % (k, t_dict[k]))
	for k in t_dict.keys():
		if not k.startswith('_'):
			smart_crop_results['t_'+k] = t_dict[k]
			print('   %-21s : %s' % (k, t_dict[k]))
			
	gc.collect()

	return VD, smart_crop_results



if __name__ == '__main__':
	import os
	import glob

	import statistics
	import matplotlib.pyplot as plt

	if False:
		import matplotlib.rcsetup as rcsetup
		print('backends', rcsetup.all_backends)
		
		print(' ')
		import matplotlib
		print('fname', matplotlib.matplotlib_fname())
		input('...')

	# test super params
	video_limit = 200
	pause = False

	print('\n\n ~~~ SmartVidCrop ~~~~')

	# check annotations
	print(' Checking "annotations" directory...')
	if not os.path.isdir(os.path.join(root_path, 'annotations')):
		print(' Error: "annotations" directory not found. ')
		print(' Please download annotations from github repository.')
	available_annots = []
	
	for annot_index in [1, 2, 3, 4, 5, 6]:
		available_annots.append('annotator_' + str(annot_index))
		c_a_dir = os.path.join(root_path, 'annotations', 'annotator_' + str(annot_index))
		if not os.path.isdir(c_a_dir):
			print(' "annotator_' + str(annot_index) + '" directory not found.')
			print(' Please download annotations from the original github repository.')
			sys.exit(0)
		else:
			continue

		zip_path = os.path.join(root_path, 'annotations', 'annotator_' + str(annot_index) + '.zip')
		if not os.path.isfile(zip_path):
			print(' ' + zip_path + ' file not found. ')
			print(' Please download files from the original github repository.')
			sys.exit(0)
		print(' extracting "annotator_' + str(annot_index) + '.zip"...')
		with zipfile.ZipFile(zip_path, 'r') as zip_ref:
			zip_ref.extractall(os.path.join(root_path, 'annotations'))

	# load annotations
	print(' loading annotations...')
	vid_inds = list(range(1, 101)) + list(range(601, 701))
	annots = []
	for i, folder in enumerate(available_annots):
		annots.append([])
		for k, mode in enumerate(['1-3', '3-1']):
			annots[i].append([])
			for j, ind in enumerate(vid_inds):
				annots[i][k].append([])
				file = '%03d_%s.txt' % (ind, mode)
				with open(os.path.join(root_path, 'annotations', folder, file)) as fp:
					lines = fp.read().splitlines()
				for l in lines:
					annots[i][k][j].append(int(l.split(',')[k]))
	print(' ...found annotations from %d users, for' % len(annots))
	for i in range(len(annots)):
		print(' %d' % len(annots[i]), end='')
	print(' target aspect ratios and ', end='')
	for i in range(len(annots)):
		print(' %d' % len(annots[i][0]), end='')
	print(' videos')

	# setup initial crop params
	crop_params_test = sc_init_crop_params()
	crop_params_test['exit_on_spread_sal'] = False
	crop_params_test['exit_on_low_cvrg'] = False
	crop_params_test['com_km'] = False
	crop_params_test['t_border'] = -1						  

	# setup tests
	print_eval_frames = False
	tests = {}					
	for t in [200]:
		for use3d in [False]:
			for min_c in [26]:
				for min_t_cut in [1.0]:
					for max_cuts in [0]:
						crop_params_test['t_threshold'] = t
						crop_params_test['use_3d'] = use3d
						crop_params_test['no_extra_cuts'] = max_cuts
						crop_params_test['hdbscan_min'] = min_c
						crop_params_test['min_t_cut'] = min_t_cut
						test_name = 't=' + str(t) + \
									'_3d=' + str(int(use3d)) + \
									'_mc=' + str(max_cuts) + \
									'_cs=' + str(min_c) + \
									'_mt=' + str(min_t_cut)
						tests[test_name] = crop_params_test.copy()
						

	print(' Tests::')
	for i, test_name in enumerate(tests.keys()):
		print(' %3d: %s' % (i + 1, str(test_name)))
	input('...')

	# paths
	vids_in_dir = os.path.join(root_path, 'DHF1k', '')
	pkl_out_dir = os.path.join(root_path, '__pkls', '')
	results_out_top = os.path.join(root_path, 'results', '')
	os.makedirs(pkl_out_dir, exist_ok=True)
	os.makedirs(results_out_top, exist_ok=True)

	# input videos
	vid_paths = glob.glob(os.path.join(vids_in_dir, '*.AVI'))
	vid_paths.sort()
	print(' Videos:: found %d videos' % min(video_limit,len(vid_paths)))
	
	# start
	for test_name in tests.keys():
		for iorp, orp in enumerate(['1:3', '3:1']):  # '4:5', '1:1'
			cur_crop_params = tests[test_name]
			cur_crop_params['out_ratio'] = orp
			for i, vid_path in enumerate(vid_paths):
				
				# check limit
				if i==video_limit:
					break
					
				# check if we already processed the session
				vid_fn = os.path.basename(vid_path).split('.')[0]
				suffix = vid_fn + '_' + str(orp.replace(':', '-'))
				if os.path.isfile(os.path.join(results_out_top, test_name, suffix+'.txt')) and\
					os.path.isfile(os.path.join(results_out_top, test_name, suffix+'_info.txt')):
					print('skipping:', test_name, suffix)
					continue
					
				if os.path.isfile(os.path.join(results_out_top, test_name, suffix+'.txt')) and\
					os.path.isfile(os.path.join(results_out_top, test_name, suffix+'_info.txt')):
					print('skipping:', test_name, suffix)
					continue
					
				# setup paths and params
				results_out = os.path.join(results_out_top, str(test_name))
				demo_fn = os.path.join(results_out, suffix + '_demo')
				demo_fn = ''
				plots_fn = os.path.join(results_out, suffix + '_plot.png')
				plots_fn = ''
				final_vid_fn=os.path.join(results_out, suffix)
				pathlib.Path(results_out).mkdir(parents=True, exist_ok=True)
				print('\n\n')
				print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				print(' video (%d/%d): %s' % (i + 1, len(vid_paths), vid_path))
				print(' test:', test_name)
				print(' results_out:', results_out)

				# smart crop method
				vid_data, info_dict = smart_vid_crop(vid_path, cur_crop_params,
												demo_fn=demo_fn,
												final_vid_fn=final_vid_fn,
												plots_fn=plots_fn,
												temp_path=temp_path,
												verbose=True, save_vid=False)

				# write report to txt file
				with open(os.path.join(results_out, suffix + '_info.txt'), 'w') as stfp:
					for k in info_dict.keys():
						stfp.write(k+':'+str(info_dict[k])+'\n')

				# write bounding boxes to txt file
				with open(os.path.join(results_out, suffix + '.txt'), 'w') as bbfp:
					for bb in vid_data['bbs']:
						bbfp.write('%d,%d,%d,%d\n' % (bb[0], bb[1], bb[2], bb[3]))

				# eval
				if not 'bbs' in vid_data.keys():
					print(' Bounding boxes are not available!')
					print(' Cannot proceed to evaluation!')
				else:
					print(' Evaluation::')
					if print_eval_frames:
						cap = cv2.VideoCapture(vid_path)
					user_evals = []
					for user in range(6):
						frames_ious = []
						for iframe, bb in enumerate(vid_data['bbs']):
							# [i users][k mode][j video][l frame]
							gt_d = annots[user][iorp][vid_inds.index(int(vid_fn))][iframe]
							if orp == '1:3':
								cw = 120
								ch = 360
								gt_bb = [gt_d, 0, gt_d + cw, ch]
								bb[2] = bb[0] + 120
								bb[3] = 360
							elif orp == '3:1':
								cw = 640
								ch = 214
								gt_bb = [0, gt_d, cw, gt_d + ch]
								bb[2] = 640
								bb[3] = bb[1] + 214

							frames_ious.append(bb_intersection_over_union(gt_bb, bb))

							if print_eval_frames:
								print(gt_bb, '-', bb)
								flag, frame = cap.read()
								frame = cv2.rectangle(frame, (gt_bb[0], gt_bb[1]), (gt_bb[2], gt_bb[3]),
													  (0, 255, 0), 1)  # green gt
								frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]),
													  (255, 0, 0), 1)  # blue result
								cv2.imshow('video', frame)
								cv2.waitKey(0)
						vid_iou = statistics.mean(frames_ious)
						user_evals.append(vid_iou)
						print('   user #%d: %.3f' % (user + 1, vid_iou))
					print('   mean   : %.3f' % (statistics.mean(user_evals)))
					print('\n Done processing video "%s" with "%s"\n' % (vid_fn, test_name))

				# clean up
				del vid_data
				del info_dict
				gc.collect()
				
				if pause:
					input('...next video?')
					



