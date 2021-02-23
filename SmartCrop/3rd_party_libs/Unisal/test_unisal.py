import os
import glob
import ntpath
import time

import unisal_handler
from pathlib import Path

import cv2
import numpy as np


folder_path = Path('D:/smart_crop/3rdp_saliency/code-Unisal(PyTorch)/demo')
is_video = False
source=None

if False:
	unisal_handler.predictions_from_folder(
			folder_path, is_video, source=None, train_id=None, model_domain=None)
			
			
			
			
else:	

	t = time.time()
	img_paths = glob.glob(os.path.join(folder_path,"*.png"))
	img_paths = img_paths + glob.glob(os.path.join(folder_path,"*.jpg"))
	img_paths = img_paths + glob.glob(os.path.join(folder_path,"*.jpeg"))
	print(' glob: %.3fs' % (time.time()-t))

	t = time.time()
	images = []
	out_names = []
	for img_path in img_paths:
		image = cv2.imread(img_path)
		image = np.ascontiguousarray(image[:, :, ::-1])
		images.append(image)
		out_names.append(ntpath.basename(img_path))
	print(' ...loaded %d images' % len(images))
	print(' ', images[0].shape)
	print(' imread: %.3fs' % (time.time()-t))
	
	print(' ')
	t = time.time()
	unisal_model_images = unisal_handler.init_unisal_for_images()
	print(' init: %.3fs' % (time.time()-t))
	
	out_dir = os.path.join(folder_path,'sal')
	try:
		os.mkdir(out_dir)
	except:
		pass
		
	print(' predictions as images + write ')
	for i in range(3):
		t = time.time()
		unisal_handler.predictions_from_memory(unisal_model_images, images, out_names, out_dir)
		print('   #%d: %.3fs / per img: %.3fs ' % (i+1, time.time()-t, (time.time()-t)/len(images)))
		
	print(' predictions as images ')
	for i in range(3):
		t = time.time()
		unisal_handler.predictions_from_memory(unisal_model_images, images, [], [])
		print('   #%d: %.3fs / per img: %.3fs ' % (i+1, time.time()-t, (time.time()-t)/len(images)))
		
		
		
	print(' ')
		
	t = time.time()
	unisal_model_frames = unisal_handler.init_unisal_for_frames()
	print(' init: %.3fs' % (time.time()-t))
		
	out_dir = os.path.join(folder_path,'sal_vid')
	try:
		os.mkdir(out_dir)
	except:
		pass
	print(' predictions as frames + write ')
	for i in range(3):
		t = time.time()
		unisal_handler.predictions_from_frames_in_memory(unisal_model_frames, images, out_names, out_dir)
		print(' pred + write #%d: %.3fs / per img: %.3fs ' % (i+1, time.time()-t, (time.time()-t)/len(images)))
		
	print(' predictions as frames ')
	for i in range(3):
		t = time.time()
		unisal_handler.predictions_from_frames_in_memory(unisal_model_frames, images, [], [])
		print(' pred #%d: %.3fs / per img: %.3fs ' % (i+1, time.time()-t, (time.time()-t)/len(images)))
	

	
	
	