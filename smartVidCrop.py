# basic imports
import copy
import math
import os
import pathlib
import sys
import numpy as np
import pickle as cPickle

# OpenCV for reading images from disk
import cv2

# utils for clustering
import hdbscan

# imutils for fast video reading
from imutils.video import FileVideoStream
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans

# for 3d resizing
from scipy.ndimage import zoom
from scipy.signal import resample_poly
from scipy.interpolate import RegularGridInterpolator

# for statistics calculation
import statistics

HDBSCAN_MIN_CLUSTER_SIZE = 24
hdbs_clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, allow_single_cluster=True)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# SciPy's interpolation
from scipy import interpolate

# SciPy's butterworth lowpass filters
from scipy import signal

# get path that the current script file is in
root_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
temp_path = None
temp_path = os.path.join(root_path, 'temp')

### 3rd party libs loading
if False:
	# DCNN-based shot segmentation using TransNetV2
	transnetv2_full_path = os.path.join(root_path, '3rd_party_libs', 'transnetv2', 'inference')
	print(' (adding path %s)' % transnetv2_full_path)
	sys.path.insert(0, transnetv2_full_path)
	print(' loading transnet v2 model')
	import transnetv2
	trans_threshold = 0.5
	transnet_model = transnetv2.TransNetV2(model_dir=os.path.join(transnetv2_full_path, 'transnetv2-weights'))
else:
	# DCNN-based shot segmentation
	transnetv1_full_path = os.path.join(root_path, '3rd_party_libs', 'transnetv1')
	print(' (adding path %s)' % transnetv1_full_path)
	sys.path.insert(0, transnetv1_full_path)
	import transnetv1_handler

	stn_params = transnetv1_handler.ShotTransNetParams()
	stn_params.CHECKPOINT_PATH = os.path.join('3rd_party_libs', 'transnetv1', 'shot_trans_net-F16_L3_S2_D256')
	trans_threshold = 0.1
	transnet_model = transnetv1_handler.ShotTransNet(stn_params, session=sess)

# DCNN-based Saliency detection using UNISAL
unisal_full_path = os.path.join(root_path, '3rd_party_libs', 'unisal')
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

# hard params
DEMO_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEMO_FONT_SCALE = 0.4
DEMO_FONT_COLOR = (255, 255, 255)
DEMO_FONT_POS = (2, 15)
RENDER_REDEMO = False


# Reads a video into a list of lists
# each list is the collection of frames for a shot
# frames are in rgb
def read_and_segment_video(video_path, shots=None):
	# open video, get info and close it
	vcap = cv2.VideoCapture(video_path)
	fr = vcap.get(cv2.CAP_PROP_FPS)
	frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
	vcap.release()
	del vcap

	# read original frames and resize frames for shot segmentation
	fvs = FileVideoStream(video_path).start()
	iii = -1
	imgs = []

	# if shot boundaries were not provided, init an array that
	# will hold the resized frames for DCNN-based shot segmenatation
	if shots is None:
		transnet_frames = np.zeros((frame_count, stn_params.INPUT_HEIGHT, stn_params.INPUT_WIDTH, 3),
									   dtype=np.uint8)

		print(' - Saliency detection...')
		t = cv2.getTickCount()
		smaps = []
		for segment_index, segment in enumerate(segments_sel):
			smaps.append(unisal_handler.predictions_from_memory_nuint8(unisal_model, segment, [], ''))
		sc_register_time(t, 'sal')
		
		
	print(' - Saliency detection...')
	t = cv2.getTickCount()
	smaps = []
	for segment_index, segment in enumerate(segments_sel):
		smaps.append(unisal_handler.predictions_from_memory_nuint8(unisal_model, segment, [], ''))
	sc_register_time(t, 'sal')
		
	while fvs.more():
		if ((iii + 1) % 59 == 0) or (iii > frame_count - 50):
			print('\r %d/%d' % (iii + 1, frame_count), end='', flush=True)
		frame = fvs.read()

		if frame is None:
			break
		iii += 1

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if shots is None:
			transnet_frames[iii, :, :, :] = cv2.resize(frame, (stn_params.INPUT_WIDTH, stn_params.INPUT_HEIGHT))
		imgs.append(frame)
	fvs.stop()
	del fvs
	print(' gathered %d frames...' % len(imgs))

	if shots is None:
		# shot segmentation
		print('\n shot segmentation...')
		trans_probs = transnet_model.predict_video(transnet_frames)
		del transnet_frames
		shots = predictions_to_scenes(trans_probs, threshold=trans_threshold)
		del trans_probs

	segments = []
	total = 0
	shots[0][0] = 0
	for shot in shots:
		i_start = shot[0]
		i_end = shot[1]
		segments.append([])
		for i in range(i_start, i_end + 1):
			if i >= len(imgs):
				break
			img = imgs[i]
			segments[-1].append(img)
			total += 1
	if (frame_count != total):
		print(' Error!')
		print(' frame_count:', frame_count)
		print(' total:', total)

		for ishot, shot in enumerate(shots):
			total = 0
			i_start = shot[0]
			i_end = shot[1]
			for i in range(i_start, i_end + 1):
				total += 1
			print(' ', ishot, ':', i_start, '-', i_end, '(', total, ')')
		# input('...')
		del imgs
		return segments, fr, frame_count, shots

	del imgs

	return segments, fr, frame_count, shots


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


# Initiates the time dictionary
sc_times_crop = {}


def sc_init_time():
	global sc_times_crop
	sc_times_crop = {}


# Registers a time in time dictionary
def sc_register_time(t, key_name):
	add_t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
	if key_name in sc_times_crop.keys():
		sc_times_crop[key_name] += add_t
	else:
		sc_times_crop[key_name] = add_t


# Prints the time dictionary
def sc_print_crop_times(vid_dur):
	total_s = ''
	sum_t = 0
	sum_p = 0
	for key_name in sc_times_crop.keys():
		sum_t += sc_times_crop[key_name]
		sum_p += (sc_times_crop[key_name] / vid_dur) * 100.0
		total_s += ' %-18s : %7.3fs, %6.3f%%\n' % (
			key_name, sc_times_crop[key_name], (sc_times_crop[key_name] / vid_dur) * 100.0)
	total_s += ' %-18s : %7.3fs, %6.3f%%\n' % ("Total", sum_t, sum_p)
	return total_s, sum_p


# Initiates the SmartCrop method's parameters to the default settings
def sc_init_crop_params(print_dict=False):
	crop_params = {}

	crop_params['force_crop'] = False

	crop_params['out_ratio'] = "4:5"
	crop_params['max_input_d'] = 250
	crop_params['skip'] = 6
	crop_params['clust_filt'] = True
	crop_params['resize_factor'] = 1.0
	crop_params['op_close'] = True
	crop_params['value_bias'] = 1.0  # bias conversion of image value to 3rd dimension for clustering
	crop_params['border_det'] = True
	crop_params['spread_sal_exit'] = True
	crop_params['select_sum'] = 1  # if 1, select cluster with max sum, else select cluster with value
	crop_params['use_3d'] = True
	crop_params['xtra_cuts'] = True

	crop_params['force_crop'] = True

	crop_params['shift_time'] = 6

	crop_params['loess_filt'] = 1
	crop_params['loess_w_secs'] = 3
	crop_params['loess_degree'] = 2

	crop_params['lp_filt'] = 1
	crop_params['lp_cutoff'] = 2
	crop_params['lp_order'] = 5

	crop_params['t_sal'] = 40  # max mean saliency to continue (if higher than this -> pad)
	crop_params['t_coverage'] = 0.60  # min coverage to continue (if lower than this -> pad)
	crop_params['t_threshold'] = 120
	crop_params['t_border'] = 10
	
	crop_params['t_cuts'] = 120
	crop_params['extra_cuts'] = 0

	if print_dict:
		for x in crop_params.keys():
			print(x, ':', crop_params[x])

	return crop_params


# Render a collection of frames to a video file
# input frames are in RGB, uses OpenCV so it writes frames in BGR
def sc_renderer(segments, fr, outPath, verbose=False):
	if outPath:
		# setup writer
		frame_height, frame_width, channels = segments[0][0].shape
		if verbose:
			print(' %dx%dx%d, fr=%.3f' % (frame_height, frame_width, channels, fr))
		fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		mp4_out = cv2.VideoWriter(outPath + '.mp4', fourcc, fr * 2.0, (frame_width, frame_height))
		print('%s' % (outPath + '.mp4'))

		# write frames
		for i, segm in enumerate(segments):
			for j in range(len(segm)):
				temp_frame = cv2.cvtColor(segm[j], cv2.COLOR_RGB2BGR)
				mp4_out.write(temp_frame)
		mp4_out.release()


#
def sc_redemo(vid_path, final_ds, conversion_mode, frame_w, frame_h, process_w, process_h, bbs, convert_to_rgb=True):
	scale_h = float(process_h) / float(frame_h)
	scale_w = float(process_w) / float(frame_w)

	# open video, get info and close it
	vc = cv2.VideoCapture(vid_path + '.mp4');
	fr = vc.get(cv2.CAP_PROP_FPS)
	fw = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
	fh = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
	vc.release()
	del vc

	# setup writer
	outPath = vid_path + '2.mp4'
	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	mp42_out = cv2.VideoWriter(outPath, fourcc, fr, (fw + process_w, fh))
	print('%s' % outPath)

	# setup reader
	fvs = FileVideoStream(vid_path + '.mp4').start()
	iii = -1
	imgs = []

	while fvs.more():
		frame = fvs.read()
		if frame is None:
			break
		iii += 1

		# get frame of video (left-most part)
		re_frame = np.copy(frame[:, 0:process_w])
		if conversion_mode == 1:
			cx = int(final_ds[iii] * scale_w)
			cy = int(fh / 2.0)
		elif conversion_mode == 2:
			cx = int(fw / 2.0)
			cy = int(final_ds[iii] * scale_h)
		re_frame = cv2.circle(re_frame, (cx, cy), 2, (0, 255, 0), -1)

		# bounding boxes have the scale of the original video and we must resize them to the demo video size
		bx1, by1, bx2, by2 = bbs[iii]
		re_frame = cv2.rectangle(re_frame,
								 (int(bx1 * scale_w), int(by1 * scale_h) - 2),
								 (int(bx2 * scale_w), int(by2 * scale_h) - 2),
								 (0, 255, 0), 1)
		cv2.putText(re_frame, 'smoothed BB', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR, 1)

		# concat and write
		out = np.concatenate((frame, re_frame), axis=1)
		if convert_to_rgb:
			out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
		mp42_out.write(out)

	fvs.stop()
	del fvs
	mp42_out.release()

	os.remove(vid_path + '.mp4')
	os.rename(vid_path + '2.mp4', vid_path + '.mp4')

#
def get_points_on_line(p1x, p1y, p2x, p2y, imageW, imageH):
	#difference and absolute difference between points
	#used to calculate slope and relative location between points
	dX = p2x - p1x
	dY = p2y - p1y
	dXa = np.abs(dX)
	dYa = np.abs(dY)

	#predefine numpy array for output based on distance between points
	itbuffer = np.empty(shape=(np.maximum(dYa,dXa),2),dtype=np.float32)
	itbuffer.fill(np.nan)

	#Obtain coordinates along the line using a form of Bresenham's algorithm
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
			itbuffer[:,0] = np.arange(p1x+1, +dXa+1)
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
	
def sc_check_for_extra_cuts(sal_img, prevX, prevY, curX, curY):
	h = sal_img.shape[0]
	w = sal_img.shape[1]
	points = get_points_on_line(prevX, prevY, curX, curY, w,h)
	
	# get saliency sum of points under forcus movement lines
	sum = 0.0
	for i in range(points.shape[0]):
		sum += float(sal_img[points[i,0], points[i,1]])
	mean_sal_jump = float(sum) / points.shape[0]
	mean_sal_jump = int(mean_sal_jump/255)
	return mean_sal_jump
	
	
# Calculates
def sc_calc_dest_size(output_ratio, orig_width, orig_height):
	# compute target width and heigth based on crop_or_pad and output_ratio
	if output_ratio == "-":
		out_w = orig_width
		out_h = orig_height
		conversion_mode = 0
		print('   preserving original aspect ratio (%d,%d)' % (orig_width, orig_height))
	else:
		c = output_ratio.split(':')
		height_ratio = float(c[0])
		width_ratio = float(c[1])
		print('   (%d:%d)' % (height_ratio, width_ratio), end='')
		# if (width_ratio*orig_width)<(height_ratio*orig_height):
		if height_ratio > width_ratio:
			print(' (portrait-to-landscape conversion preserving width)', end='')
			conversion_mode = 2
			out_w = orig_width
			out_h = int(math.floor((width_ratio / height_ratio) * orig_width))
		else:
			print(' (landscape-to-portrait preserving height)', end='')
			conversion_mode = 1
			out_w = int(math.floor((height_ratio / width_ratio) * orig_height))
			out_h = orig_height

	return out_w, out_h, conversion_mode


def sc_compute_bb(crop_params, final_ds, conversion_mode, frame_w, frame_h, process_w, process_h, bb_w, bb_h):
	scale_h = float(process_h) / float(frame_h)
	scale_w = float(process_w) / float(frame_w)

	# scale center coordinates back to original dimensions
	bb_h = int(bb_h / scale_h)
	bb_w = int(bb_w / scale_w)
	final_xs = [0] * len(final_ds)
	final_ys = [0] * len(final_ds)
	if conversion_mode == 1:
		for i in range(len(final_ds)):
			final_xs[i] = int(final_ds[i] / scale_w)
			final_ys[i] = 0
	else:
		for i in range(len(final_ds)):
			final_xs[i] = 0
			final_ys[i] = int(final_ds[i] / scale_h)


	bbs = []
	for i in range(len(final_ds)):
		# compute final bounding box around center coords
		x1 = int(final_xs[i]) - int(math.floor(bb_w / 2.0))
		y1 = int(final_ys[i]) - int(math.floor(bb_h / 2.0))
		x2 = int(final_xs[i]) + int(math.ceil(bb_w / 2.0))
		y2 = int(final_ys[i]) + int(math.ceil(bb_h / 2.0))

		# ensure bounding box has given dimensions
		x1 += (x2 - x1) - bb_w
		y1 += (y2 - y1) - bb_h

		# ensure bounding box is in frame
		if x1 < 0:
			x1 = 0
			x2 = bb_w
		if x2 > frame_w:
			x1 = frame_w - bb_w
			x2 = frame_w
		if y1 < 0:
			y1 = 0
			y2 = bb_h
		if y2 > frame_h:
			y1 = frame_h - bb_h
			y2 = frame_h

		# register bounding box
		bbs.append([x1, y1, x2, y2])
	return bbs, bb_w, bb_h


def sc_border_detection(crop_params, segments):
	# input frames are in RGB
	start_t = cv2.getTickCount()
	orig_h, orig_w, orig_channs = segments[0][0].shape
	print(' (h:', orig_h, ', w:', orig_w, ')')

	# mean
	f_mean = np.zeros((orig_h, orig_w), np.float)
	ti = 0
	for segment in segments:
		skipping = 0
		for frame in segment:
			if skipping == 0:
				ti += 1
				f_mean += cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
			skipping += 1
			if skipping > crop_params['skip']:
				skipping = 0
	f_mean = np.divide(f_mean, ti).astype(int)

	# std
	f_std = np.zeros((orig_h, orig_w), np.float)
	ti = 0
	for segment in segments:
		skipping = 0
		for frame in segment:
			if skipping == 0:
				ti += 1
				f_std += np.power(np.absolute(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) - f_mean), 2)
			skipping += 1
			if skipping > crop_params['skip']:
				skipping = 0
	f_std = np.sqrt(np.divide(f_std, ti)).astype(int)

	# compute top and bottom borders
	f_col = np.average(f_std, axis=1)
	t = 0
	for i in range(orig_h):
		if f_col[i] > crop_params['t_border']:
			break
		t += 1
	b = orig_h
	for i in range(orig_h):
		if f_col[-(i + 1)] > crop_params['t_border']:
			break
		b -= 1

	# compute left and right borders
	f_row = np.average(f_std, axis=0)
	l = 0
	for i in range(orig_w):
		if f_row[i] > crop_params['t_border']:
			break
		l += 1
	r = orig_w
	for i in range(orig_w):
		if f_row[-(i + 1)] > crop_params['t_border']:
			break
		r -= 1

	if t > orig_h / 4:
		t = int(orig_h / 4)
	if b < orig_h * (3 / 4):
		b = int(orig_h * (3 / 4))
	if l > orig_w / 4:
		l = int(orig_w / 4)
	if r < orig_w * (3 / 4):
		r = int(orig_w * (3 / 4))

	print(' (t:', t, ', b:', b, ', l:', l, ', r:', r, ')')
	if t != 0 or b != orig_h or l != 0 or r != orig_w:
		# crop frames
		for segment in segments:
			for i, frame in enumerate(segment):
				segment[i] = frame[t:b, l:r, :]

	sc_register_time(start_t, 'border_detection')
	return segments, t, orig_h - b, l, orig_w - r


def sc_find_center_of_mass(total_sal, factor=2.0, bias=1.0):
	# resize for faster operation
	initH = total_sal.shape[0]
	initW = total_sal.shape[1]
	total_sal = cv2.resize(total_sal, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_NEAREST)

	# find max val and its indicies as the initial cluster center
	max_val = np.amax(total_sal)
	[max_row, max_col] = np.unravel_index(total_sal.argmax(), total_sal.shape)

	# init & gather points
	coo = coo_matrix(total_sal).tocoo()
	X = np.vstack((coo.row, coo.col, coo.data)).transpose().astype(float)
	max_dim = max([initH / factor, initW / factor])

	# cluster
	if X.shape[0] > 0:
		X[:, 2] = (X[:, 2] / np.amax(X[:, 2])) * max_dim * bias
		X = X.astype(np.uint8)
		clusterer = KMeans(n_clusters=1, random_state=0,
						   init=np.array([[max_row, max_col, max_val]]),
						   n_init=1,
						   max_iter=5).fit(X)
	else:
		return None, None

	# return scaled back
	return clusterer.cluster_centers_[0][1] * factor, clusterer.cluster_centers_[0][0] * factor


def sc_3d_resize(O, factor):
	type_of_resize = 3

	if type_of_resize == 0:
		O = zoom(O, (factor, factor, 1))
	elif type_of_resize == 1:
		if factor < 1.0:
			factors = [(1, int(1.0 / factor)), (1, int(1.0 / factor)), (1, 1)]
		else:
			factors = [(int(factor), 1), (int(factor), 1), (1, 1)]
		for k in range(3):
			O = resample_poly(O, factors[k][0], factors[k][1], axis=k)
	elif type_of_resize == 2:
		steps = [factor, factor, 1]  # original step sizes
		x, y, z = [steps[k] * np.arange(O.shape[k]) for k in range(3)]  # original grid
		f = RegularGridInterpolator((x, y, z), O)  # interpolator
		dx, dy, dz = 1.0, 1.0, 1.0  # new step sizes
		new_grid = np.mgrid[0:x[-1]:dx, 0:y[-1]:dy, 0:z[-1]:dz]  # new grid
		new_grid = np.moveaxis(new_grid, (0, 1, 2, 3), (3, 0, 1, 2))  # reorder axes for evaluation
		O = f(new_grid)
	elif type_of_resize == 3:
		nh = int(O.shape[0] * factor)
		nw = int(O.shape[1] * factor)
		l = O.shape[2]
		On = np.zeros((nh, nw, l), dtype='uint8')
		for i in range(l):
			On[:, :, i] = cv2.resize(O[:, :, i], None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
		return On

	return O


def sc_clustering_filt_3d(sal3d, factor=2.0, select_sum=1, bias=1.0, close=True, fig=None):
	# resize for faster operations
	oh = int(sal3d.shape[0])
	ow = int(sal3d.shape[1])
	nh = int(oh / factor)
	nw = int(ow / factor)
	d = sal3d.shape[2]
	sal3d_resized = np.zeros((nh, nw, d), dtype='uint8')
	for i in range(d):
		sal3d_resized[:, :, i] = cv2.resize(sal3d[:, :, i], (nw, nh), interpolation=cv2.INTER_LINEAR)

	# init & gather points
	X = []
	W = []
	inds = np.argwhere(sal3d_resized > 0)
	for ind in inds:
		X.append([ind[0], ind[1], ind[2], sal3d_resized[ind[0], ind[1], ind[2]]])
		W.append(X[-1][3])
	l = len(X)
	max_dim = max(nh, nw)
	X = np.array(X)
	X[:, 3] = ((X[:, 3] / np.amax(X[:, 3])) * max_dim * bias)
	X = X.astype(np.uint8)
	W = np.array(W).transpose()

	if l > HDBSCAN_MIN_CLUSTER_SIZE + 1:
		# cluster
		labels = hdbs_clusterer.fit_predict(X)
		
		if fig is not None:
			# plot before
			ax = fig.add_subplot(221, projection='3d')
			sh = ax.scatter(X[:, 1], X[:, 2], X[:, 0], marker="s", c=labels, s=40, cmap="RdBu")
			ax.set_xlabel('x')
			ax.set_ylabel('t')
			ax.set_zlabel('y')
			ax.set_xlim((0, sal3d_resized.shape[1]))
			ax.set_ylim((0, sal3d_resized.shape[2]))
			ax.set_zlim((0, sal3d_resized.shape[0]))
			lg = ax.legend(*sh.legend_elements(), loc='best', title="Clusters")
			ax.grid(True)


		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
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
				per_t_cls[i].sort()
			sc_register_time(t, 'clust_per_t')

			# filter
			sal3d_resized_backup = np.copy(sal3d_resized)
			t = cv2.getTickCount()
			selX = []
			for i in range(len(X)):
				if labels[i] != max_cl:
					sal3d_resized[X[i][0], X[i][1], X[i][2]] = 0
				else:
					selX.append([X[i][0], X[i][1], X[i][2]])
			selX = np.array(selX)
			sc_register_time(t, 'clust_filtering')
			
			# check for empty sal maps
			for i in range(d):
				empty_str=' '
				if np.sum(sal3d_resized[:,:,i])==0:
					empty_str='*'
				print('%3d %s' % (i,empty_str), per_t_cls[i])

			# morphological close
			t = cv2.getTickCount()
			closing_kernel = np.ones((5, 5), np.uint8)
			if close:
				for i in range(d):
					sal3d_resized[:, :, i] = cv2.morphologyEx(np.squeeze(sal3d_resized[:, :, i]), cv2.MORPH_CLOSE,
															  closing_kernel)
			sc_register_time(t, 'clust_closing')

		if fig is not None:
			# plot after
			ax2 = fig.add_subplot(223, projection='3d')
			sh = ax2.scatter(selX[:, 1], selX[:, 2], selX[:, 0], marker="s", s=40, cmap="RdBu")
			ax2.set_xlabel('x')
			ax2.set_ylabel('t')
			ax2.set_zlabel('y')
			ax2.set_xlim((0, sal3d_resized.shape[1]))
			ax2.set_ylim((0, sal3d_resized.shape[2]))
			ax2.set_zlim((0, sal3d_resized.shape[0]))
			ax2.grid(True)

		
	# resize back
	sal3d = np.zeros((oh, ow, d), dtype='uint8')
	for i in range(d):
		sal3d[:, :, i] = cv2.resize(sal3d_resized[:, :, i], (ow, oh), interpolation=cv2.INTER_LINEAR)
		
	return sal3d


def sc_clustering_filt(total_sal, factor=2.0, select_sum=1, bias=1.0, close=True):
	# resize for faster operation
	initH = total_sal.shape[0]
	initW = total_sal.shape[1]
	total_sal = cv2.resize(total_sal, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_LINEAR)

	# init & gather points
	coo = coo_matrix(total_sal).tocoo()
	X = np.vstack((coo.row, coo.col, coo.data)).transpose()
	max_dim = max([initH / factor, initW / factor])
	X[:, 2] = ((X[:, 2] / np.amax(X[:, 2])) * max_dim * bias).astype(np.uint8)
	W = coo.data.transpose()

	if X.shape[0] > HDBSCAN_MIN_CLUSTER_SIZE + 1:
		# cluster
		labels = hdbs_clusterer.fit_predict(X)

		# calculate weight of each cluster
		n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
		if n_clusters > 0:
			weights = [0] * n_clusters
			for i in range(n_clusters):
				inds = list(np.where(labels == i)[0])
				if select_sum == 1:
					weights[i] = np.sum(W[inds])
				else:
					weights[i] = np.amax(W[inds])
			max_cl = weights.index(max(weights))

			# filter
			for i in range(len(X)):
				if labels[i] != max_cl:
					total_sal[X[i][0], X[i][1]] = 0

			# morphological close
			if close:
				t = cv2.getTickCount()
				closing_kernel = np.ones((5, 5), np.uint8)
				total_sal = cv2.morphologyEx(total_sal, cv2.MORPH_CLOSE, closing_kernel)
				sc_register_time(t, 'closing')

	# scale back image
	total_sal = cv2.resize(total_sal, (initW, initH), interpolation=cv2.INTER_LINEAR)

	return total_sal


# Auxiliary function to check if we must processs the frame with index 'index' when skip step is 'skip'
def sc_to_process(skip, index):
	if skip == 0:
		return True
	else:
		return (index % skip == 0)


def sc_compute_crop(crop_params, segments, vid_fn, demo_fn='', demo_fr=20):  # accepts RGB, writes demo in BGR

	# compute final output
	orig_h, orig_w, orig_channs = segments[0][0].shape
	pre_scale = int((crop_params['max_input_d'] / max(orig_h, orig_w)) * 100) / 100
	process_w = int(orig_w * pre_scale)
	process_h = int(orig_h * pre_scale)
	final_w, final_h, conversion_mode = sc_calc_dest_size(crop_params['out_ratio'],
														  process_w,
														  process_h)
	if conversion_mode == 1:
		final_dim = final_w
		process_dim = process_w
	else:
		final_dim = final_h
		process_dim = process_h

	# init
	segments_ds = []
	segments_ts = []
	segments_id = []
	shot_boundaries = []
	true_ts = []

	# demo video writer
	if demo_fn:
		demo_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		demo_height = process_h
		demo_width = process_w * 5
		demo_out = cv2.VideoWriter(demo_fn + '.mp4', demo_fourcc,
								   demo_fr / (crop_params['skip'] + 1),
								   (demo_width, demo_height))

	# count total frames
	fc = 0
	for segment_index, segment in enumerate(segments):
		for frame_index, frame in enumerate(segment):
			fc += 1
	print("   (oh:%d,ow:%d)->(ph:%d,pw:%d)->(fh:%d,fw:%d)" % (orig_h, orig_w, process_h, process_w, final_h, final_w))

	# sample, resize image
	t = cv2.getTickCount()
	t_i = -1
	segments_sel = []
	for segment_index, segment in enumerate(segments):
		segments_sel.append([])
		segments_ts.append([])
		true_ts.append([])
		for frame_index, frame in enumerate(segment):
			t_i += 1
			shot_boundaries.append(1 if frame_index == 0 else 0)
			if sc_to_process(crop_params['skip'], frame_index):
				segments_id.append(segment_index)
				segments_sel[segment_index].append(cv2.resize(frame, (process_w, process_h),
															  interpolation=cv2.INTER_NEAREST))
				segments_ts[segment_index].append(t_i)
			true_ts[segment_index].append(t_i)
	sc_register_time(t, 'resize')


	# check if saliency maps are already saved
	smaps = None
	if temp_path is not None:
		sal_fn = os.path.join(temp_path, vid_fn +'_sal.pkl')
		if os.path.isfile(sal_fn):
			with open(sal_fn,'rb') as fp:
				smaps = cPickle.load(fp)


	# UNISAL saliency detection
	if smaps is None:	
		print(' - Saliency detection...')
		t = cv2.getTickCount()
		smaps = []
		for segment_index, segment in enumerate(segments_sel):
			smaps.append(unisal_handler.predictions_from_memory_nuint8(unisal_model, segment, [], ''))
		sc_register_time(t, 'sal')
		
	# save sal maps
	if temp_path is not None:
		sal_fn = os.path.join(temp_path, vid_fn +'_sal.pkl')
		if not os.path.isfile(sal_fn):
			with open(sal_fn, 'wb') as fp:
				cPickle.dump(smaps, fp)
				

	if not crop_params['force_crop']:
		# check mean saliency and decide whether to continue
		mean_sal = []
		for segment_index, segment in enumerate(segments_sel):
			for frame_index, frame in enumerate(segment):
				mean_sal.append(np.average(smaps[segment_index][frame_index]))
		mean_sal_score = statistics.mean(mean_sal)
		if crop_params['spread_sal_exit'] and not crop_params['force_crop']:
			if mean_sal_score > crop_params['t_sal']:
				print('   (mean saliency: %.3f > %.3f - skipping...)' % (mean_sal_score, crop_params['t_sal']))
				return segments_ds, segments_ts, final_w, final_h, \
					   mean_sal_score, \
					   0, \
					   true_ts, shot_boundaries, process_w, process_h
		print('   (mean saliency: %.3f < %.3f -  continuing...)' % (mean_sal_score, crop_params['t_sal']))
	else:
		mean_sal_score = 0.0

	# loop through each segments frames
	cvrg_scores = []
	extra_cuts_at = []
	extra_cuts_scores = []
	ti = 0
	zero_maps = 0
	prev_dx = None
	prev_dy = None
	for segment_index, segment in enumerate(segments_sel):
		ds = []
		for frame_index, frame in enumerate(segment):
			# threshold saliency map
			t = cv2.getTickCount()
			total_sal = np.copy(smaps[segment_index][frame_index])
			total_sal[total_sal < crop_params['t_threshold']] = 0
			sc_register_time(t, 'thresh')

			# compute best possible coverage score
			if not crop_params['force_crop']:
				t = cv2.getTickCount()
				if conversion_mode == 1:
					total_sal_flat = np.sum(total_sal, axis=0).reshape(1, process_w)
				else:
					total_sal_flat = np.sum(total_sal, axis=1).reshape(1, process_h)
				t_sum = np.sum(total_sal_flat)
				max_cvrg = 0
				for d in range(total_sal_flat.shape[1] - final_dim):
					b_sum = np.sum(total_sal_flat[0, d:(d + final_dim)])
					current_cvrg = b_sum / t_sum
					if current_cvrg > max_cvrg:
						max_cvrg = current_cvrg
				cvrg_scores.append(max_cvrg)
				sc_register_time(t, 'coverage')
			else:
				cvrg_scores.append(1.0)

			# select main focus on saliency image, through clustering
			if crop_params['clust_filt']:
				t = cv2.getTickCount()
				total_sal_clust = sc_clustering_filt(total_sal,
													 factor=crop_params['resize_factor'],
													 select_sum=crop_params['select_sum'],
													 bias=crop_params['value_bias'],
													 close=crop_params['op_close'])
				sc_register_time(t, 'clustering')
			else:
				total_sal_clust = total_sal_flat

			# get center of mass
			t = cv2.getTickCount()
			dx, dy = sc_find_center_of_mass(total_sal_clust, 
											factor=crop_params['resize_factor'],
											bias=crop_params['value_bias'])
			if conversion_mode == 1:
				d = dx
			else:
				d = dy
			if d is None:
				if len(ds) > 0:
					d = ds[-1]
				else:
					d = int(process_dim / 2)
				dx = int(process_w/2)
				dy = int(process_h/2)

			sc_register_time(t, 'center_of_mass')
	
			# check if we must inset a cut
			t = cv2.getTickCount()
			if crop_params['extra_cuts']>0:
				if prev_dx is not None:
					mean_sal_jump = sc_check_for_extra_cuts(total_sal, prev_dx, prev_dy, dx, dy)
					if mean_sal_jump<crop_params['t_cut']:
						extra_cuts_at.append(ti)
						extra_cuts_scores.append(mean_sal_jump)
				prev_dx = dx
				prev_dy = dy
			sc_register_time(t, 'extra_cuts')
			
			# write demo video
			if demo_fn:
				out_frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
				total_sal_u = cv2.cvtColor(smaps[segment_index][frame_index], cv2.COLOR_GRAY2BGR)
				total_sal = cv2.cvtColor(total_sal, cv2.COLOR_GRAY2BGR)
				total_sal_clust = cv2.cvtColor(total_sal_clust, cv2.COLOR_GRAY2BGR)

				total_overlayed = np.copy(out_frame_bgr)
				cv2.addWeighted(total_sal_clust, 0.5, total_overlayed, 1 - 0.5, 0, total_overlayed)

				cv2.putText(total_sal_u, 'saliency', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR, 1)
				cv2.putText(total_sal, 'post-processed', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR, 1)
				cv2.putText(total_sal_clust, 'clustering', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR,
							1)
				cv2.putText(total_overlayed, 'overlayed', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR, 1)

				total_sal_clust = cv2.circle(total_sal_clust, (int(dx), int(dy)), 3, (0, 255, 0), -1)
				total_overlayed = cv2.circle(total_overlayed, (int(dx), int(dy)), 3, (0, 255, 0), -1)
				temp_frame_out = np.concatenate(
					(out_frame_bgr, total_sal_u, total_sal, total_sal_clust, total_overlayed), axis=1)
				demo_out.write(temp_frame_out)

			# register signals
			ds.append(d)
			ti += 1

		segments_ds.append(ds)

	if demo_fn:
		demo_out.release()

	return segments_ds, segments_ts, final_w, final_h, \
		   mean_sal_score, \
		   statistics.mean(cvrg_scores), \
		   true_ts, shot_boundaries, process_w, process_h, conversion_mode, \
		   extra_cuts_at, extra_cuts_scores


def sc_compute_crop_3d(crop_params, segments, vid_fn, demo_fn='', demo_fr=20, fig=None):  # accepts RGB, writes demo in BGR

	# compute final output
	orig_h, orig_w, orig_channs = segments[0][0].shape
	process_w = int(orig_w * (crop_params['max_input_d'] / max(orig_h, orig_w)))
	while (process_w % 4 != 0):
		process_w += 1
	process_h = int(orig_h * (crop_params['max_input_d'] / max(orig_h, orig_w)))
	while (process_h % 4 != 0):
		process_h += 1
	final_w, final_h, conversion_mode = sc_calc_dest_size(crop_params['out_ratio'],
														  process_w,
														  process_h)
	if conversion_mode == 1:
		final_dim = final_w
		process_dim = process_w
	else:
		final_dim = final_h
		process_dim = process_h

	# init
	segments_ds = []
	segments_ts = []
	segments_id = []
	shot_boundaries = []
	true_ts = []

	# count total frames taking into consideration skip
	fc = 0
	fcs = 0
	for segment_index, segment in enumerate(segments):
		for frame_index, frame in enumerate(segment):
			fc += 1
			if sc_to_process(crop_params['skip'], frame_index):
				fcs += 1
	print("   (oh:%d,ow:%d)->(ph:%d,pw:%d)->(fh:%d,fw:%d)" % (orig_h, orig_w, process_h, process_w, final_h, final_w))

	# sample, resize image, saliency detection and resize
	t_i = -1


	# check if saliency maps are already saved
	smaps = None
	if temp_path is not None:
		sal_fn = os.path.join(temp_path, vid_fn +'_sal.pkl')
		if os.path.isfile(sal_fn):
			with open(sal_fn,'rb') as fp:
				smaps = cPickle.load(fp)


	if smaps is None:	
		smaps = np.zeros((process_h, process_w, fcs), dtype=np.uint8)
		segmentations = []
		last_end = 0
		for segment_index in range(len(segments)):
			segments_ts.append([])
			true_ts.append([])
			
			# skip and resize segemnts
			t = cv2.getTickCount()
			print(' - Resize... ')
			process_ind = -1
			for frame_index in range(len(segments[segment_index])):
				t_i += 1
				true_ts[segment_index].append(t_i)
				shot_boundaries.append(1 if frame_index == 0 else 0)
				if sc_to_process(crop_params['skip'], frame_index):
					process_ind += 1
					segments_id.append(segment_index)
					segments_ts[segment_index].append(t_i)
					segments[segment_index][process_ind] = cv2.resize(segments[segment_index][frame_index],
																	  (process_w, process_h),
																	  interpolation=cv2.INTER_NEAREST)
			sc_register_time(t, 'resize')

			# delete not needed frames
			for del_i in range(len(segments[segment_index]) - 1, process_ind, -1):
				del segments[segment_index][del_i]
				
			# compute segmentation taking into account skipped frames
			ind_start = last_end
			ind_end = last_end + len(segments[segment_index]) 
			last_end += len(segments[segment_index])
			segmentations.append([ind_start, ind_end])
			
			# saliency detection
			t = cv2.getTickCount()
			print(' - Saliency detection... ')
			smaps[:, :, ind_start:ind_end] = unisal_handler.predictions_from_memory_nuint8_np(unisal_model,
														segments[segment_index], [], '')
			if not demo_fn:
				segments[segment_index] = []  # free up memory now that we have the segments salinecy maps
			sc_register_time(t, 'saliency_detection')

	# save sal maps
	if temp_path is not None:
		sal_fn = os.path.join(temp_path, vid_fn +'_sal.pkl')
		if not os.path.isfile(sal_fn):
			with open(sal_fn, 'wb') as fp:
				cPickle.dump(smaps, fp)
			
	# check mean saliency
	if not crop_params['force_crop']:
		t = cv2.getTickCount()
		mean_sal = []
		for iii in range(fcs):
			mean_sal.append(np.average(smaps[:, :, iii]))
		mean_sal_score = statistics.mean(mean_sal)
		if crop_params['spread_sal_exit'] and not crop_params['force_crop']:
			if mean_sal_score > crop_params['t_sal']:
				print('   (mean saliency: %.3f > %.3f - skipping...)' % (mean_sal_score, crop_params['t_sal']))
				return segments_ds, segments_ts, final_w, final_h, \
					   mean_sal_score, \
					   0, \
					   0, \
					   0, \
					   true_ts, shot_boundaries, process_w, process_h
		print('   (mean saliency: %.3f < %.3f -  continuing...)' % (mean_sal_score, crop_params['t_sal']))
		sc_register_time(t, 'check_mean_saliency')
	else:
		mean_sal_score = 0.0

	# threshold
	if demo_fn:
		smaps_copy = np.copy(smaps)
	t = cv2.getTickCount()
	print(' - Thresholding...')
	smaps[smaps < crop_params['t_threshold']] = 0
	sc_register_time(t, 'thresh')

	# compute best possible coverage score
	cvrg_scores = []
	if not crop_params['force_crop']:
		t = cv2.getTickCount()
		print(' - Coverage score calculation... ')
		for iii in range(fcs):
			if conversion_mode == 1:
				total_sal_flat = np.sum(smaps[:, :, iii], axis=0).reshape(1, process_w)
			else:
				total_sal_flat = np.sum(smaps[:, :, iii], axis=1).reshape(1, process_h)
			t_sum = np.sum(total_sal_flat)
			max_cvrg = 0.0
			for d in range(total_sal_flat.shape[1] - final_dim):
				b_sum = np.sum(total_sal_flat[0, d:(d + final_dim)])
				current_cvrg = b_sum / t_sum
				if current_cvrg > max_cvrg:
					max_cvrg = current_cvrg
			cvrg_scores.append(max_cvrg)
		print('%.3fs' % ((cv2.getTickCount() - t) / cv2.getTickFrequency()))
		sc_register_time(t, 'center_of_mass')
	else:
		for iii in range(fcs):
			cvrg_scores.append(1.0)

	# clustering
	if demo_fn:
		smaps_thresh = np.copy(smaps)
	t = cv2.getTickCount()
	print(' - Clustering... ')
	for segment in segmentations:
		smaps[:, :, segment[0]:segment[1]] = sc_clustering_filt_3d(smaps[:, :, segment[0]:segment[1]],
																   factor=crop_params['resize_factor'],
																   select_sum=crop_params['select_sum'],
																   bias=crop_params['value_bias'],
																   close=crop_params['op_close'],
																   fig=fig)										   
	sc_register_time(t, 'clustering')
			
	# demo video writer
	if demo_fn:
		demo_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
		demo_height = process_h
		demo_width = process_w * 5
		demo_out = cv2.VideoWriter(demo_fn + '.mp4',
								   demo_fourcc,
								   demo_fr / (crop_params['skip'] + 1),
								   (demo_width, demo_height))

	# get center of mass
	print(' - Center of mass... ')
	iii = 0
	prev_dx = None
	prev_dy = None
	extra_cuts_at = []
	extra_cuts_scores = []
	for segment_index, segment in enumerate(segments):
		ds = []
		for frame_index in range(len(segment)):
			t = cv2.getTickCount()
			if np.sum(smaps[:, :, iii]) > 0:
				dx, dy = sc_find_center_of_mass(smaps[:, :, iii],
												factor=crop_params['resize_factor'],
												bias=crop_params['value_bias'])
			else:
				print(' empty sal map at %d pos %d (%d)' % (segment_index, frame_index, iii))
				dx = None
				dy = None
			iii += 1

			if conversion_mode == 1:
				d = dx
			else:
				d = dy
			if d is None:
				if len(ds) > 0:
					d = ds[-1]
				else:
					d = int(process_dim / 2)
				dx = int(process_w/2)
				dy = int(process_h/2)
			sc_register_time(t, 'center_of_mass')
			
			# check if we must inset a cut
			t = cv2.getTickCount()
			if crop_params['extra_cuts']>0:
				if prev_dx is not None:
					mean_sal_jump = sc_check_for_extra_cuts(smaps[:, :, iii], prev_dx, prev_dy, dx, dy)
					if mean_sal_jump<crop_params['t_cut']:
						extra_cuts_at.append(iii)
						extra_cuts_scores.append(mean_sal_jump)
				prev_dx = dx
				prev_dy = dy
			sc_register_time(t, 'extra_cuts')

			# register signals
			ds.append(d)

			# write demo video
			if demo_fn:
				out_frame_bgr = cv2.cvtColor(segments[segment_index][frame_index], cv2.COLOR_RGB2BGR)
				total_sal_raw = cv2.cvtColor(smaps_copy[:, :, frame_index], cv2.COLOR_GRAY2BGR)
				total_sal_threshed = cv2.cvtColor(smaps_thresh[:, :, frame_index], cv2.COLOR_GRAY2BGR)
				total_sal_clust = cv2.cvtColor(smaps_copy[:, :, frame_index], cv2.COLOR_GRAY2BGR)

				total_overlayed = np.copy(out_frame_bgr)
				cv2.addWeighted(total_sal_clust, 0.5, total_overlayed, 1 - 0.5, 0, total_overlayed)

				cv2.putText(total_sal_raw, 'saliency', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR, 1)
				cv2.putText(total_sal_threshed, 'post-processed', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE,
							DEMO_FONT_COLOR, 1)
				cv2.putText(total_sal_clust, 'clustering', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR,
							1)
				cv2.putText(total_overlayed, 'overlayed', DEMO_FONT_POS, DEMO_FONT, DEMO_FONT_SCALE, DEMO_FONT_COLOR, 1)

				total_sal_clust = cv2.circle(total_sal_clust, (int(dx), int(dy)), 3, (0, 255, 0), -1)
				total_overlayed = cv2.circle(total_overlayed, (int(dx), int(dy)), 3, (0, 255, 0), -1)
				temp_frame_out = np.concatenate(
					(out_frame_bgr, total_sal_raw, total_sal_threshed, total_sal_clust, total_overlayed), axis=1)
				demo_out.write(temp_frame_out)

		segments_ds.append(ds)

	if demo_fn:
		demo_out.release()

	return segments_ds, segments_ts, final_w, final_h, \
		   mean_sal_score, \
		   statistics.mean(cvrg_scores), \
		   true_ts, shot_boundaries, process_w, process_h, conversion_mode, \
		   extra_cuts_at, extra_cuts_scores


def sc_interpolate(segments_ds, segments_ts, true_ts):
	# init interpolated segments and total series
	segments_dsi = []
	segments_tsi = []

	# interpolate each segment separately
	for segment_index in range(len(segments_ts)):
		segments_tsi.append(true_ts[segment_index])
		try:
			fi = interpolate.interp1d(segments_ts[segment_index], 
									segments_ds[segment_index], 
									fill_value="extrapolate",
									kind="quadratic")
			segments_dsi.append(fi(true_ts[segment_index]))
		except:
			segments_dsi.append( np.array([float(np.mean(np.array(segments_ds[segment_index])))] * len(true_ts[segment_index])))



	return segments_dsi, segments_tsi


def sc_butter_lowpass_filter(x, cutoff, fs, order):
	try:
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
		try:
			return signal.filtfilt(b, a, x), False
		except:
			pass
	except:
		pass

	y = np.convolve(x, np.ones(5), 'same') / 5
	for i in range(2, len(x) - 2):
		x[i] = y[i]
	return x, True


def sc_smoothing(segments_dsi, segments_tsi, fr, loess_filt, window_to_fr, degree, lp_filt, lp_cutoff, lp_order):
	final_ds = []

	for i in range(len(segments_tsi)):
		errd = False
		segment_start = segments_tsi[i][0]
		l = len(segments_dsi[i])
		t_vec = np.array(list(range(0, l)))
		segm_d = np.array(segments_dsi[i])

		# adjust window based on segment length
		adj_window = min(int(fr * window_to_fr), l)
		if (adj_window % 2) == 0:
			adj_window -= 1

		
		# low pass
		if lp_filt:
			segm_lp, errd = sc_butter_lowpass_filter(segm_d, lp_cutoff, fr * 2.0, lp_order)
		else:
			segm_lp = segm_d

		# loess
		if loess_filt:
			loess = pyloess.Loess(t_vec, segm_lp)
			# loess = sm.nonparametric.lowess(segm_lp, t_vec)[:,0]
			segm_sm = []
			try:
				for j in range(len(segments_dsi[i])):
					segm_sm.append(loess.estimate(j, window=adj_window, use_matrix=False, degree=degree))

				if np.isnan(np.sum(segm_sm)):
					segm_sm=list(segm_lp)
			except:
				base = np.zeros_like(segm_d)
				for e in range(len(segm_sm)):
					base[e] += segm_sm[e]
				print('segm_d', segm_d)
				print('segm_lp', segm_lp)
				print('t_vec', t_vec)
				print('base', base)
				print('len(segm_d)', len(segm_d))
				print('len(segm_lp)', len(segm_lp))
				print('len(base)', len(base))

				plt.plot(t_vec, segm_d, 'r')  # plotting t, a separately
				plt.plot(t_vec, segm_lp, 'b')  # plotting t, b separately
				plt.plot(t_vec, base, 'g', linestyle=':')  # plotting t, a separately
				plt.show()
				input('... loess failed')
			final_ds = final_ds + segm_sm
		else:
			for j in range(len(segments_dsi[i])):
				final_ds.append(segments_dsi[j])

	return final_ds


def sc_plot_signals(segments_dsi, final_ds, true_ts, shot_boundaries, fig):
	# concat signals to a single list
	orig_ds = []
	for s in segments_dsi:
		for y in s:
			orig_ds.append(y)
	ats = []
	for s in true_ts:
		for t in s:
			ats.append(t)
	shot_lines = [x * max(orig_ds) for x in shot_boundaries]

	# plot signals
	axs1 = fig.add_subplot(222)
	pld, = axs1.plot(ats, orig_ds, color=(0, 0.5, 0.7))
	axs1.plot(ats, shot_lines, color=(0, 0, 0))
	axs1.set_xlim(-1, len(orig_ds))
	axs1.set_ylim(1, max(orig_ds))

	for item in ([axs1.title, axs1.xaxis.label, axs1.yaxis.label] +
				 axs1.get_xticklabels() + axs1.get_yticklabels()):
		item.set_fontsize(6)
		
	axs2 = fig.add_subplot(224)
	plfd, = axs2.plot(ats, final_ds, color=(0.5, 0.0, 0.7))
	axs2.plot(ats, shot_lines, color=(0, 0, 0))
	axs2.set_xlim(-1, len(final_ds))
	axs2.set_ylim(1, max(final_ds))
	
	for item in ([axs2.title, axs2.xaxis.label, axs2.yaxis.label] +
				 axs2.get_xticklabels() + axs2.get_yticklabels()):
		item.set_fontsize(6)





def sc_crop_transform(segments, transform_type, bbs, bb_w, bb_h, verbose=False, rgb2bgr=False):
	### bounding boxes are in [x1,y1,x2,y2]
	t = cv2.getTickCount()

	# transform frames
	if transform_type == 1:
		iii = -1
		for i, segm in enumerate(segments):
			for j in range(len(segm)):
				iii += 1

				x1 = bbs[iii][0]
				y1 = bbs[iii][1]
				x2 = bbs[iii][2]
				y2 = bbs[iii][3]

				crop_img = cv2.resize(segm[j][y1:y2, x1:x2, :], (bb_w, bb_h))
				if rgb2bgr:
					crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
				segm[j] = crop_img
	elif transform_type == 2:
		for i, segm in enumerate(segments):
			for j in range(len(segm)):
				pad_img = cv2.copyMakeBorder(segm[j], int(math.ceil(bb_h / 2)), int(math.floor(bb_h / 2)),
											 int(math.ceil(bb_w / 2)), int(math.floor(bb_w / 2)),
											 cv2.BORDER_CONSTANT, value=[0, 0, 0])
				if rgb2bgr:
					pad_img = cv2.cvtColor(pad_img, cv2.COLOR_RGB2BGR)
				segm[j] = pad_img

	sc_register_time(t, 'transform')
	return segments


def smart_crop(crop_params, segments, orig_fr, 
				demo_fn='', final_vid_fn='', plots_fn='',
				convert_to_rgb=False):
	# clear registered time measurements
	sc_init_time()
	smart_crop_results = {}
	do_pad = False

	# ensure proper crop_params
	if crop_params is None:
		crop_params = sc_init_crop_params

	# ensure RGB input
	if convert_to_rgb:
		for i in range(len(segments)):
			for j in range(len(segments[i])):
				segments[i][j] = cv2.cvtColor(segments[i][j], cv2.COLOR_BGR2RGB)



	# init figure
	if plots_fn:
		fig = plt.figure()
	else:
		fig = None
		 
	# get basic info
	segments_backup_local = copy.deepcopy(segments)
	orig_h, orig_w, orig_c = segments[0][0].shape
	fc = 0
	for temp_segm in segments_backup_local:
		fc += len(temp_segm)

	# process segments
	if not crop_params['use_3d']:
		print(' - Processing...')
		segments_ds, segments_ts, final_w, final_h, \
		mean_sal_score, cvrg_score, \
		true_ts, shot_boundaries, \
		process_w, process_h, \
		conversion_mode, \
		extra_cuts_at, extra_cuts_scores = \
			sc_compute_crop(crop_params, segments, vid_fn, 
								demo_fn=demo_fn, demo_fr=orig_fr)
	else:
		print(' - Processing 3D...')
		segments_ds, segments_ts, final_w, final_h, \
		mean_sal_score, cvrg_score, \
		true_ts, shot_boundaries, \
		process_w, process_h, \
		conversion_mode, \
		extra_cuts_at, extra_cuts_scores = \
			sc_compute_crop_3d(crop_params, segments, vid_fn, 
								demo_fn=demo_fn, demo_fr=orig_fr, 
								fig=fig)  # ALL segments coordinates are downscaled by pre_scale
		
		
	# insert extra cuts
	if crop_params['extra_cuts']>0:
		t = cv2.getTickCount()
		extra_cuts_at = [x for _,x in sorted(zip(extra_cuts_scores,extra_cuts_at))]
		extra_cuts_scores.sort()
		del extra_cuts_at[:crop_params['extra_cuts']]
		del extra_cuts_scores[:crop_params['extra_cuts']]
		
		segments_dsa = []
		segments_tsa = []
		iii=0
		for i in range(len(segments_ds)):
			segment_dsa_temp = []
			segment_tsa_temp = []	
			for j in range(len(segment)):
				segment_dsa_temp.append(segments_ds[i][j])
				segment_tsa_temp.append(segments_ts[i][j])
				if iii in extra_cuts_at:
					segments_dsa.append(segment_dsa_temp)
					segments_tsa.append(segment_tsa_temp)
					segment_dsa_temp = []
					segment_tsa_temp = []
				iii+=1
			segments_dsa.append(segment_dsa_temp)
			segments_tsa.append(segment_tsa_temp)
		sc_register_time(t, 'extra_cuts_ins')
		
	# check mean saliency
	if mean_sal_score > crop_params['t_sal'] and crop_params['spread_sal_exit']:
		if not crop_params['force_crop']:
			do_pad = True
	else:
		# check coverage score and decide if we continue
		if cvrg_score < crop_params['t_coverage'] and not crop_params['force_crop']:
			do_pad = True
		else:

			# interpolate
			print(' - Interpolating...')
			t = cv2.getTickCount()
			segments_dsi, segments_tsi = \
				sc_interpolate(segments_ds, segments_ts, true_ts)
			sc_register_time(t, 'interpolation')
			

			# smooth signals
			print(' - Smoothing...')
			t = cv2.getTickCount()
			final_ds = sc_smoothing(segments_dsi, segments_tsi, orig_fr,
									crop_params['loess_filt'], crop_params['loess_w_secs'], crop_params['loess_degree'],
									crop_params['lp_filt'], crop_params['lp_cutoff'], crop_params['lp_order'])
			sc_register_time(t, 'loess+lp')
			

			# plot
			if plots_fn:
				print(' - Plotting series...')
				sc_plot_signals(segments_dsi,
								final_ds,
								true_ts,
								shot_boundaries,
								fig)
				plt.savefig(plots_fn)		
				

			# compute bounding boxes from center coordinates
			print(' - Computing BB...')
			t = cv2.getTickCount()
			bbs, bb_w, bb_h = sc_compute_bb(crop_params, final_ds, conversion_mode,
											orig_w, orig_h,
											process_w, process_h,
											final_w, final_h)
			sc_register_time(t, 'bb')
			# ALL segments coordinates are now scaled back to original size
			# bounding in boxes are in [x1,y1,x2,y2]

			# shift time series backwards to catch up fast movements
			t = cv2.getTickCount()
			if crop_params['shift_time'] > 0:
				for i in range(crop_params['shift_time']):
					bbs[-i] = bbs[-crop_params['shift_time']]
				for i in range(len(bbs) - crop_params['shift_time']):
					bbs[i] = bbs[i + crop_params['shift_time']]
			sc_register_time(t, 'shift')

			# render complete demo
			if RENDER_REDEMO:
				if demo_fn:
					print(' - Rendering complete demo at ', end='')
					sc_redemo(demo_fn, final_ds, conversion_mode,
							  orig_w, orig_h,
							  process_w, process_h,
							  bbs, convert_to_rgb=convert_to_rgb)
			transform_type = 1  # smart-crop

			smart_crop_results['results'] = 'smart cropped'
			print('   (proceeding to smart-crop)')

	if do_pad:
		c = crop_params['out_ratio'].split(':')
		height_ratio = float(c[0])
		width_ratio = float(c[1])
		if (width_ratio * orig_w) > (height_ratio * orig_h):  # pad height
			bb_w = 0
			bb_h = int(math.floor((width_ratio / height_ratio) * orig_w)) - orig_h
		else:  # pad width
			bb_w = int(math.floor((height_ratio / width_ratio) * orig_h)) - orig_w
			bb_h = 0
		bbs = []
		transform_type = 2  # pad
		smart_crop_results['results'] = 'padded'
		print('   (proceeding to pad)')

	# transforming frames
	print(' - Transforming...')
	segments_backup_local = sc_crop_transform(segments_backup_local, transform_type, bbs, bb_w, bb_h)

	# render final video
	if final_vid_fn:
		print(' - Rendering final video at ', end='')
		sc_renderer(segments_backup_local, orig_fr, final_vid_fn)

	# write info text
	info_string = ' --- Profile::\n'
	info_string += ' (%d,%d)->(%d,%d)\n' % (orig_h, orig_w, bb_h, bb_w)
	info_string += ' %-18s : %7d\n' % ('max_input_dim', crop_params['max_input_d'])
	info_string += ' %-18s : %7d\n' % ('clustering', crop_params['clust_filt'])
	info_string += ' %-18s : %7d\n' % ('threshold', crop_params['t_threshold'])
	info_string += ' --- Times::\n'
	temp_s, sum_p = sc_print_crop_times(fc / orig_fr)
	info_string += temp_s
	info_string += ' --- Scores::\n'
	info_string += ' %-18s : %7.3f \n' % ('mean saliency', mean_sal_score)
	info_string += ' %-18s : %7.3f \n' % ('coverage score', cvrg_score)
	info_string += ' %-18s : %s\n' % ('--- Result', smart_crop_results['results'])
	info_string += ' --- Parameters::\n'
	for cp in crop_params.keys():
		info_string += ' %-18s : %s\n' % (cp, str(crop_params[cp]))

	# results json
	smart_crop_results['original_dim'] = '(%d,%d)' % (orig_h, orig_w)
	smart_crop_results['result_dim'] = '(%d,%d)' % (bb_h, bb_w)
	smart_crop_results['mean_sal_score'] = mean_sal_score
	smart_crop_results['mean_sal_score_t'] = crop_params['t_sal']
	smart_crop_results['coverage_score'] = cvrg_score
	smart_crop_results['coverage_score_t'] = crop_params['t_coverage']
	smart_crop_results['percentage'] = sum_p

	# clean up
	del segments, segments_ds, segments_ts
	del final_ds, final_w, final_h
	del shot_boundaries

	# ensure BGR output
	if convert_to_rgb:
		for i in range(len(segments_backup_local)):
			for j in range(len(segments_backup_local[i])):
				segments_backup_local[i][j] = cv2.cvtColor(segments_backup_local[i][j], cv2.COLOR_RGB2BGR)

	return segments_backup_local, info_string, smart_crop_results, sc_times_crop, bbs


def smart_crop_version():
	return '1.0'


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



	print('\n\n ~~~ SmartVidCrop ~~~~')

	# check annotations
	print(' Checking "annotations" directory...')
	if not os.path.isdir(os.path.join(root_path, 'annotations')):
		print(' Error: "annotations" directory not found. Please download annotations from github repository.')
	available_annots = []
	
	for annot_index in [1, 2, 3, 4, 5, 6]:
		available_annots.append('annotator_' + str(annot_index))
		if not os.path.isdir(os.path.join(root_path, 'annotations', 'annotator_' + str(annot_index))):
			print(' "annotator_' + str(
				annot_index) + '" directory not found. Please download annotations from the original github repository.')
			sys.exit(0)
		else:
			continue

		zip_filepath = os.path.join(root_path, 'annotations', 'annotator_' + str(annot_index) + '.zip')
		if not os.path.isfile(zip_filepath):
			print(' ' + zip_filepath + ' file not found. Please download files from the original github repository.')
			sys.exit(0)
		print(' extracting "annotator_' + str(annot_index) + '.zip"...')
		with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
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
	print(' target aspect raitons and ', end='')
	for i in range(len(annots)):
		print(' %d' % len(annots[i][0]), end='')
	print(' videos')

	# setup initial crop params
	crop_params_test = sc_init_crop_params()
	crop_params_test['force_crop'] = True

	# setup tests
	print_eval_frames = False
	tests = {}
	for t in [150]:
		for use3d in [False, True]:
			for max_cuts in [10,0]:
				crop_params_test['t_threshold'] = t
				crop_params_test['use_3d'] = use3d
				crop_params_test['extra_cuts'] = max_cuts
				test_name = 't=' + str(t) + '_3d=' + str(int(use3d)) + '_mc=' + str(max_cuts)
				tests[test_name] = crop_params_test
	print(' Tests::')
	for i, test_name in enumerate(tests.keys()):
		print(' %3d: %s' % (i + 1, str(test_name)))


	# paths
	vids_in_dir = './DHF1k/'
	pkl_out_dir = './__pkls/'
	results_out_top = './results/'
	os.makedirs(pkl_out_dir, exist_ok=True)
	os.makedirs(results_out_top, exist_ok=True)

	# input videos
	vid_paths = glob.glob(os.path.join(vids_in_dir, '*.AVI'))
	#vid_paths = ['./DHF1k/004.AVI']
	video_limit = 1000
	print(' Videos:: found %d videos' % min(video_limit,len(vid_paths)))


	# start
	for test_name in tests.keys():
		for iorp, orp in enumerate(['1:3', '3:1']):  # '4:5', '1:1'
			crop_params_test = tests[test_name]
			crop_params_test['out_ratio'] = orp
			for i, vid_path in enumerate(vid_paths):
				# check limit
				if i==video_limit:
					break

				# setup paths and params
				vid_fn = os.path.basename(vid_path).split('.')[0]
				suffix = vid_fn + '_' + str(orp.replace(':', '-'))
				results_out = os.path.join(results_out_top, str(test_name))
				demo_fn = os.path.join(results_out, suffix + '_demo')
				plots_fn = os.path.join(results_out, suffix + '_plot.png')
				final_vid_fn=os.path.join(results_out, suffix)
				pathlib.Path(results_out).mkdir(parents=True, exist_ok=True)
				print('\n\n')
				print(' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				print(' video (%d/%d): %s' % (i + 1, len(vid_paths), vid_path))
				print(' profile:', suffix)
				print(' results_out:', results_out)

				# read original video

				shots = None
				if temp_path is not None:
					shots_fn = os.path.join(temp_path, vid_fn +'_shots.pkl')
					if os.path.isfile(shots_fn):
						with open(shots_fn,'rb') as fp:
							shots = cPickle.load(fp)
				print(' - Reading original video %s...' % vid_fn)
				segments, fr_test, fc_test, shots = read_and_segment_video(vid_path, shots)
				if temp_path is not None:
					if not os.path.isfile(shots_fn):
						with open(shots_fn, 'wb') as fp:
							cPickle.dump(shots, fp)
							
				# main smart crop method
				segments_backup, total_s, results_dict, sc_times_crop, bbs = smart_crop(crop_params_test, segments,
																						fr_test,
																						demo_fn=demo_fn,
																						final_vid_fn=final_vid_fn,
																						plots_fn=plots_fn)

				# write report to txt file
				with open(os.path.join(results_out, suffix + '_info.txt'), 'w') as stfp:
					stfp.write(total_s)

				# write bounding boxes to txt file
				with open(os.path.join(results_out, suffix + '.txt'), 'w') as bbfp:
					for bb in bbs:
						bbfp.write('%d,%d,%d,%d\n' % (bb[0], bb[1], bb[2], bb[3]))

				# eval
				print(' Evaluation::')
				if print_eval_frames:
					cap = cv2.VideoCapture(vid_path)
				user_evals = []
				for user in range(6):
					frames_ious = []
					for iframe, bb in enumerate(bbs):
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
					print(' user #%d: %.3f' % (user + 1, vid_iou))
				print(' mean   : %.3f' % (statistics.mean(user_evals)))

				print('\n Done processing video "%s" with "%s"\n' % (vid_fn, test_name))
				plt.close('all')







