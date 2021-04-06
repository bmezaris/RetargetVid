import ffmpeg
import numpy as np
import tensorflow as tf
import cv2

transnet_verbose = False

class ShotTransNetParams:
	F = 16
	L = 3
	S = 2
	D = 256
	INPUT_WIDTH = 48
	INPUT_HEIGHT = 27
	CHECKPOINT_PATH = None

class ShotTransNet:

	def __init__(self, params: ShotTransNetParams, session=None):
		self.params = params
		self.session = session or tf.Session()
		self._build()
		self._restore()

	def _build(self):
		def shape_text(tensor):
			return ", ".join(["?" if i is None else str(i) for i in tensor.get_shape().as_list()])

		with self.session.graph.as_default():
			if transnet_verbose:
				print(" [ShotTransNet] Creating ops.")

			with tf.variable_scope("TransNet"):
				def conv3d(inp, filters, dilation_rate):
					return tf.keras.layers.Conv3D(filters, kernel_size=3, dilation_rate=(dilation_rate, 1, 1),
												  padding="SAME", activation=tf.nn.relu, use_bias=True,
												  name="Conv3D_{:d}".format(dilation_rate))(inp)

				self.inputs = tf.placeholder(tf.uint8,
											 shape=[None, None, self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3])
				net = tf.cast(self.inputs, dtype=tf.float32) / 255.
				if transnet_verbose:
					print(" " * 10, "Input ({})".format(shape_text(net)))

				for idx_l in range(self.params.L):
					with tf.variable_scope("SDDCNN_{:d}".format(idx_l + 1)):
						filters = (2 ** idx_l) * self.params.F
						if transnet_verbose:
							print(" " * 10, "SDDCNN_{:d}".format(idx_l + 1))

						for idx_s in range(self.params.S):
							with tf.variable_scope("DDCNN_{:d}".format(idx_s + 1)):
								net = tf.identity(net)  # improves look of the graph in TensorBoard
								conv1 = conv3d(net, filters, 1)
								conv2 = conv3d(net, filters, 2)
								conv3 = conv3d(net, filters, 4)
								conv4 = conv3d(net, filters, 8)
								net = tf.concat([conv1, conv2, conv3, conv4], axis=4)
								if transnet_verbose:
									print(" " * 10, "> DDCNN_{:d} ({})".format(idx_s + 1, shape_text(net)))

						net = tf.keras.layers.MaxPool3D(pool_size=(1, 2, 2))(net)
						if transnet_verbose:
							print(" " * 10, "MaxPool ({})".format(shape_text(net)))

				shape = [tf.shape(net)[0], tf.shape(net)[1], np.prod(net.get_shape().as_list()[2:])]
				net = tf.reshape(net, shape=shape, name="flatten_3d")
				if transnet_verbose:
					print(" " * 10, "Flatten ({})".format(shape_text(net)))
				net = tf.keras.layers.Dense(self.params.D, activation=tf.nn.relu)(net)
				if transnet_verbose:
					print(" " * 10, "Dense ({})".format(shape_text(net)))

				self.logits = tf.keras.layers.Dense(2, activation=None)(net)
				if transnet_verbose:
					print(" " * 10, "Logits ({})".format(shape_text(self.logits)))
				self.predictions = tf.nn.softmax(self.logits, name="predictions")[:, :, 1]
				if transnet_verbose:
					print(" " * 10, "Predictions ({})".format(shape_text(self.predictions)))

			print(" [ShotTransNet] Network built.")
			no_params = np.sum([int(np.prod(v.get_shape().as_list())) for v in tf.trainable_variables()])
			print(" [ShotTransNet] Found {:d} trainable parameters.".format(no_params))

	def _restore(self):
		if self.params.CHECKPOINT_PATH is not None:
			saver = tf.train.Saver()
			saver.restore(self.session, self.params.CHECKPOINT_PATH)
			print(" [ShotTransNet] Parameters restored from '{}'.".format(self.params.CHECKPOINT_PATH))

	def predict_raw(self, frames: np.ndarray):
		assert len(frames.shape) == 5 and \
			   list(frames.shape[2:]) == [self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3],\
			" [ShotTransNet] Input shape must be [batch, frames, height, width, 3]."
		return self.session.run(self.predictions, feed_dict={self.inputs: frames})

	def predict_video(self, frames: np.ndarray):
		assert len(frames.shape) == 4 and \
			   list(frames.shape[1:]) == [self.params.INPUT_HEIGHT, self.params.INPUT_WIDTH, 3], \
			" [ShotTransNet] Input shape must be [frames, height, width, 3]."

		def input_iterator():
			# return windows of size 100 where the first/last 25 frames are from the previous/next batch
			# the first and last window must be padded by copies of the first and last frame of the video
			no_padded_frames_start = 25
			no_padded_frames_end = 25 + 50 - (len(frames) % 50 if len(frames) % 50 != 0 else 50)  # 25 - 74

			start_frame = np.expand_dims(frames[0], 0)
			end_frame = np.expand_dims(frames[-1], 0)
			padded_inputs = np.concatenate(
				[start_frame] * no_padded_frames_start + [frames] + [end_frame] * no_padded_frames_end, 0
			)

			ptr = 0
			while ptr + 100 <= len(padded_inputs):
				out = padded_inputs[ptr:ptr+100]
				ptr += 50
				yield out

		res = []
		for inp in input_iterator():
			pred = self.predict_raw(np.expand_dims(inp, 0))[0, 25:75]
			res.append(pred)
			#print("\r shot segmentation... {}/{}".format(
			#	min(len(res) * 50, len(frames)), len(frames)
			#), end="", flush=True)
		return np.concatenate(res)[:len(frames)]  # remove extra padded frames

		
def shot_preprocess_frame(frame):
	return cv2.resize(frame, (ShotTransNetParams.INPUT_WIDTH, ShotTransNetParams.INPUT_HEIGHT))
	

def shot_preprocess_frames(frames):
	l = frames.shape[0]
	shottransnet_frames = np.zeros((l, ShotTransNetParams.INPUT_HEIGHT, ShotTransNetParams.INPUT_WIDTH, 3))
	for i in range(l):
		shottransnet_frames[i,:,:,:] = cv2.resize(frames[i,:,:,:], (ShotTransNetParams.INPUT_WIDTH, ShotTransNetParams.INPUT_HEIGHT))
	return shottransnet_frames
		

def shot_preprocess_frames_list(frames):
	l = len(frames)
	shottransnet_frames = np.zeros((l, ShotTransNetParams.INPUT_HEIGHT, ShotTransNetParams.INPUT_WIDTH, 3))
	for i in range(l):
		shottransnet_frames[i,:,:,:] = cv2.resize(frames[i], (ShotTransNetParams.INPUT_WIDTH, ShotTransNetParams.INPUT_HEIGHT))
	return shottransnet_frames		
		
		
		
		
		
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

def find_closest_larger(x, s):
	for i in s:
		if i>x:
			return i
	for i in s:
		if i>=x:
			return i
	return s[-1]

	
def smooth(x, window):
	w=np.ones(window,'d')
	y=np.convolve(w/w.sum(), x, mode='same')
	return y
	
def find_extremas(x, order):
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
	
def process_sd_x(x, window=3, order=9, verbose=False):
	l = x.shape[0]
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
	
	return y
	
def assert_segmentation(shots, l, min_frames=12):
	# ensure no shots with length < "min_frames" frames
	exclude_indices = []
	for i in range(len(shots)):
		if shots[i][1]-shots[i][0]<12:
			exclude_indices.append(i)
	if len(exclude_indices)>0:
		for i in sorted(exclude_indices, reverse=True):
			del shots[i]
			
	# ensure at least 1 shot (the entire video)
	if len(shots)==0:
		shots.append(0, l-1)
		
	# ensure the old format where shot boundaries were neighboring frames
	for i in range(len(shots)-1):
		if shots[i][1]!=(shots[i+1][0]-1):
			shots[i][1] = shots[i+1][0]-1
		
	# ensure last shot reaches the end of the video
	if shots[-1][1]<l-1:
		shots[-1][1]=l-1
		
	return shots
		
def shots_from_predictions(predictions: np.ndarray, threshold: float = 0.1):
	predictions = (predictions > threshold).astype(np.uint8)

	shots = []
	t, tp, start = -1, 0, 0
	for i, t in enumerate(predictions):
		if tp == 1 and t == 0:
			start = i
		if tp == 0 and t == 1 and i != 0:
			shots.append([start, i])
		tp = t
	if t == 0:
		shots.append([start, i])
		
	shots = assert_segmentation(shots, len(predictions), min_frames=12)

	return np.array(shots, dtype=np.int32)

def shots_from_predictions_extended(predictions: np.ndarray, threshold: float = 0.1):
	predictions_post = process_sd_x(predictions)
	shots = []
	t, tp, start = -1, 0, 0
	for i, t in enumerate(predictions_post):
		if tp == 1 and t == 0:
			start = i
		if tp == 0 and t == 1 and i != 0:
			shots.append([start, i])
		tp = t
	if t == 0:
		shots.append([start, i])
	return shots


def shot_trans_net_load_and_predict(video_path):
	video_stream, err = (
		ffmpeg
		.input(video_path)
		.output('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(params.INPUT_WIDTH, params.INPUT_HEIGHT))
		.run(capture_stdout=True)
	)
	video = np.frombuffer(video_stream, np.uint8).reshape([-1, params.INPUT_HEIGHT, params.INPUT_WIDTH, 3])
	predictions = net.predict_video(video)
	shots = scenes_from_predictions(predictions, threshold=0.1)
	return shots
	
def shot_trans_net_handler_version():
	return '1.0'
	
	