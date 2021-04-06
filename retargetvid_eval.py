import os
import statistics
import numpy as np
import sys
import zipfile

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


# get path that the current script file is in
root_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# check annotations
print(' Checking "annotations" directory...')
if not os.path.isdir(os.path.join(root_path, 'annotations')):
	print(' Error: "annotations" directory not found. Please download annotations from github repository.')
available_annots = []

for annot_index in [1, 2, 3, 4, 5, 6]:
	zip_filepath = os.path.join(root_path, 'annotations', 'annotator_' + str(annot_index) + '.zip')
	annot_path = os.path.join(root_path, 'annotations', 'annotator_' + str(annot_index))
	available_annots.append('annotator_' + str(annot_index))
	
	if not os.path.isdir(annot_path):
		if not os.path.isfile(zip_filepath):
			print(' "annotator_' + str(
				annot_index) + '" directory and zip not found. Please download annotations from the original github repository.')
			sys.exit(0)
	else:
		continue

	if not os.path.isfile(zip_filepath):
		print(' ' + zip_filepath + ' file not found. Please download files from the original github repository.')
		sys.exit(0)
	print(' extracting "annotator_' + str(annot_index) + '.zip"...')
	with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
		zip_ref.extractall(os.path.join(root_path, 'annotations'))

# load annotations
# NOTE: annots structure [user]{ar}{video}[frame]
print(' loading annotations...')
vid_inds = list(range(1, 101)) + list(range(601, 701))
annots = []
for i, folder in enumerate(available_annots):
	annots.append([])
	annots[i] = {}
	for ar in ['1-3', '3-1']:
		d_select = (0 if ar=='1-3' else 1)
		annots[i][ar] = {}
		for j, vid_ind in enumerate(vid_inds):
			file = '%03d_%s.txt' % (vid_ind, ar)
			with open(os.path.join(root_path, 'annotations', folder, file)) as fp:
				lines = fp.read().splitlines()
				
			annots[i][ar][vid_ind] = []
			for l in lines:
				annots[i][ar][vid_ind].append(int(l.split(',')[d_select]))
print(' ...found annotations from %d users, for' % len(annots))
for i in range(len(annots)):
	if not len(annots[i].keys())==2:
		print('Error in annotations apsect ratios count.\n Please redownload from the original GitHub repository.')
	if not len(annots[i]['1-3'].keys())==200:
		print('Error in annotations videos count.\n Please redownload from the original GitHub repository.')
	if not len(annots[i]['3-1'].keys())==200:
		print('Error in annotations videos count.\n Please redownload from the original GitHub repository.')
	
# scan for runs in the results folder
runs_folder = 'results'
runs = [os.path.split(f)[-1] for f in os.scandir(runs_folder) if f.is_dir()]
runs.sort()

# get frames count for each vid
# NOTE: annots structure [user]{ar}{video}[frame]
frame_counts = {}
for i,vid_ind in enumerate(vid_inds):
	frame_counts[vid_ind] = len(annots[0]['1-3'][1])
	
# check validity of each run in the result folder
print(' Checking runs validity...')
for run in runs:
	file_errors_count = 0
	frame_count_errors_count = 0
	for vid_ind in vid_inds:
		for ar in ['1-3','3-1']: 
			fn = os.path.join(root_path, runs_folder, run, '%03d_%s.txt' % (vid_ind, ar))
			if not os.path.isfile(fn):
				file_errors_count += 1
			else:
				with open(fn, 'r') as fp:
					lines = fp.read().splitlines()
				if abs(frame_counts[vid_ind]-len(lines))>10:
					# print(os.path.split(fn)[-1],':',frame_counts[vid_ind],'-', len(lines))
					frame_count_errors_count+=1
	print(' - %-24s (errors=%d+%d)' % (run,file_errors_count,frame_count_errors_count))



# process each run
print(' Processing runs...')
evals = {}
stats_dict = {}
missing_files = {}
for run in runs:
	missing_files[run] = 0
	evals[run] = {}
	stats_dict[run] = {}
	for ar in ['1-3','3-1']: 
		if ar not in stats_dict[run].keys():
			stats_dict[run][ar] = {}
		if ar not in evals[run].keys():
			evals[run][ar] = []
			evals[run][ar].append([])
			evals[run][ar].append([])
			evals[run][ar].append([])
			evals[run][ar].append([])
			evals[run][ar].append([])
			evals[run][ar].append([])
		
		for vid_ind in vid_inds:
			fn = os.path.join(root_path, runs_folder, run, '%03d_%s.txt' % (vid_ind, ar))
			if not os.path.isfile(fn):
				missing_files[run] += 1
				continue
			with open(fn) as fp:
				bbs_raw = fp.read().splitlines()
			bbs=[]
			for tbb in bbs_raw:
				c = tbb.split(',')
				bbs.append([int(c[0]), int(c[1]), int(c[2]), int(c[3])])
				
			for user in range(6):
				frames_ious=[]
				for iframe,method_bb in enumerate(bbs):
					# NOTE: annots structure [user]{ar}{video}[frame]
					annot_gt = annots[user][ar][vid_ind][iframe]
					if ar=='1-3':
						cw = 120
						ch = 360
						annot_bb = [annot_gt,0,annot_gt+cw,ch]
						method_bb[2] = method_bb[0]+120
						method_bb[3] = 360
					else:
						cw = 640
						ch = 214
						annot_bb = [0,annot_gt,cw,annot_gt+ch]
						method_bb[2] = 640
						method_bb[3] = method_bb[1] + 214

					frames_ious.append(bb_intersection_over_union(annot_bb, method_bb))
					
				vid_iou = statistics.mean(frames_ious)
				evals[run][ar][user].append(vid_iou)
		
			# try loading statistics from info.txt file
			fn = os.path.join(root_path, runs_folder, run, '%03d_%s_info.txt' % (vid_ind, ar))
			if os.path.isfile(fn):
				with open(fn) as fp:
					info_raw = fp.read().splitlines()
				for k in info_raw:
					if '%' in k:
						id = k.split(':')[0].strip().lower()
						val = float(k.split(',')[1].replace('%','').strip())
						if id not in stats_dict[run][ar].keys():
							stats_dict[run][ar][id] = []
						stats_dict[run][ar][id].append(val)
	
print('\n Evaluation:')
print(' %-12s %-5s %-5s %-5s %4s %3s  %-5s %-5s %-5s %4s %3s' % ('Method', 'Worst','Best', 'Mean', 'tm', 'ta', 'Worst', 'Best', 'Mean', 'tm', 'ta', ))
print(' %-12s %-26s  %-26s' % ('', '{:^26s}'.format("1:3"), '{:^26s}'.format("3:1"),))

for run in evals.keys():

	print(' %-12s' % (run), end='')
	for ar in evals[run].keys():
		users_ious = []
		for user in range(6):
			users_ious.append(statistics.mean(evals[run][ar][user]))
		best_score = max(users_ious)
		worst_score = min(users_ious)
		mean_score = statistics.mean(users_ious)
		print(' %5.1f %5.1f %5.1f ' % (worst_score*100,best_score*100,mean_score*100), end='')

		if 'total' in stats_dict[run][ar].keys():
			if len(stats_dict[run][ar]['total'])>2:
				print('%4d %3d ' % (int(max(stats_dict[run][ar]['total'])), int(statistics.mean(stats_dict[run][ar]['total']))), end='')
				continue
		print('%4s %3s '  % ('', ''), end='')

					
	if False:
		for ar in evals[run].keys():
			for id in stats_dict[run][ar].keys():
				if len(stats_dict[run][ar][id])>2:
					print('%5d %6.3f,  med:%6.3f,  +/-:%6.3f)' % (str(id),
						min(stats_dict[run][ar][id]),
						max(stats_dict[run][ar][id]),
						statistics.mean(stats_dict[run][ar][id]),
						np.median(stats_dict[run][ar][id]),
						statistics.stdev(stats_dict[run][ar][id])) )
					
		for ar in evals[run].keys():		
			print(' %-12s (missing annotations: %d)' % ('', missing_files[run]))

	print('')

	
