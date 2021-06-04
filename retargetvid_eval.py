import os
import statistics
import numpy as np
import sys
import zipfile
import time

verbose_checking = False
verbose_resuts = False

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
		annots[i][ar] = {}
		for j, vid_ind in enumerate(vid_inds):
			file = '%03d_%s.txt' % (vid_ind, ar)
			with open(os.path.join(root_path, 'annotations', folder, file)) as fp:
				lines = fp.read().splitlines()
			annots[i][ar][vid_ind] = []
			for l in lines:
				c = l.split(',')
				annots[i][ar][vid_ind].append([int(c[0]), int(c[1]), int(c[2]), int(c[3])])
print(' ...found annotations from %d users' % len(annots))
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
for vid_ind in vid_inds:
	frame_counts[vid_ind] = len(annots[0]['1-3'][vid_ind])
	
# check validity of each run in the result folder
runs_valid = []
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
				if abs(frame_counts[vid_ind]-len(lines))>1:
					#print(os.path.split(fn)[-1],':',frame_counts[vid_ind],'-', len(lines))
					frame_count_errors_count+=1
	print(' - %-30s (file errors:%d + frame count errors:%d)' % (run,file_errors_count,frame_count_errors_count))
	if file_errors_count==0:
		runs_valid.append(run)
print(' valid runs::')
runs = runs_valid
for run in runs:
	print(' - %s' % run)


# process each run
print(' Processing runs...')
evals = {}
stats_dict = {}
missing_files = {}
for i_run,run in enumerate(runs):
	missing_files[run] = 0
	evals[run] = {}
	stats_dict[run] = {}
	start_time = time.time()
	print(' %3d/%3d:'%(i_run+1,len(runs)), run, ' ', end='')
	for ar in ['1-3','3-1']: 
		
		if ar not in stats_dict[run].keys():
			stats_dict[run][ar] = {}
		if ar not in evals[run].keys():
			evals[run][ar] = []
			for user in range(6):
				evals[run][ar].append([])
		print('(', ar, ') ', end='')
		
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
				for iframe in range(frame_counts[vid_ind]):
					gt_bb = None
					method_bb = None
					try:
						gt_bb = annots[user][ar][vid_ind][iframe]
					except:
						if verbose_checking:
							print('  could not find ground-truth annot! ',
								  '  user:%d,ar:%s,video:%d,frame:%d' % (user,ar,vid_ind,iframe))
						missing_files[run] += 1
						continue
						
					try:
						method_bb = bbs[iframe]
					except:
						if verbose_checking:
							print('  could not find annotation! ',
							      '  run:%s,ar:%s,video:%d,frame:%d' % (run,ar,vid_ind,iframe))
						missing_files[run] += 1
						continue
							  
					if gt_bb is None:
						if verbose_checking:
							print(' Ground-truth bounding box is None...' % run)
						missing_files[run] += 1
						continue
						
					if method_bb is None:
						if verbose_checking:
							print(' Method (%s) bounding box is None...')
						missing_files[run] += 1
						continue
					
					### FIX: zero possible negative values that may occur 
					### (necessary for evaluating early implementations)
					method_bb[0] = method_bb[0] if method_bb[0]>0 else 0
					method_bb[1] = method_bb[1] if method_bb[1]>0 else 0
					method_bb[2] = method_bb[2] if method_bb[2]>0 else 0
					method_bb[3] = method_bb[3] if method_bb[3]>0 else 0
					gt_bb[0] = gt_bb[0] if gt_bb[0]>0 else 0
					gt_bb[1] = gt_bb[1] if gt_bb[1]>0 else 0
					gt_bb[2] = gt_bb[2] if gt_bb[2]>0 else 0
					gt_bb[3] = gt_bb[3] if gt_bb[3]>0 else 0

					try:
						frames_ious.append(bb_intersection_over_union(gt_bb, method_bb))
					except:
						missing_files[run] += 1
						continue
					
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
						
					elif 'cuts_clust:' in k:
						if 'cuts_clust' not in stats_dict[run][ar].keys():
							stats_dict[run][ar]['cuts_clust'] = []
						stats_dict[run][ar]['cuts_clust'].append(int(k.split(':')[1].strip()))
						
					elif 'cuts_extra:' in k:
						if 'cuts_extra' not in stats_dict[run][ar].keys():
							stats_dict[run][ar]['cuts_extra'] = []
						stats_dict[run][ar]['cuts_extra'].append(int(k.split(':')[1].strip()))
						
					elif 'no_extra_cuts:' in k:
						if 'no_extra_cuts' not in stats_dict[run][ar].keys():
							stats_dict[run][ar]['no_extra_cuts'] = []
						stats_dict[run][ar]['no_extra_cuts'].append(int(k.split(':')[1].strip()))
						
	print(' ---> %.3fs' % (time.time() - start_time))
	
# evaluation results output
print('\n Evaluation:')
if verbose_resuts:
	s_info_out = ('%-30s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s,%-6s' % \
				 ('Method',
				  'Worst','Best','Mean', 'ttm','tta','tcm','tca','ccm','cca','ecm','eca',
				  'Worst','Best','Mean', 'ttm','tta','tcm','tca','ccm','cca','ecm','eca', 'mf'))
	print(s_info_out)
	fp.write(s_info_out+'\n')

	for run in evals.keys():
		run_name = run.replace('_',',')
		s_info_out = '%-30s,' % (run_name.replace('_',','))
		for ar in evals[run].keys():
			users_ious = []
			for user in range(6):
				users_ious.append(statistics.mean(evals[run][ar][user]))
			best_score = max(users_ious)*100
			worst_score = min(users_ious)*100
			mean_score = statistics.mean(users_ious)*100

			t_total_max = -1
			t_total_avg = -1
			if 't_total' in stats_dict[run][ar].keys():
				t_total_max = max(stats_dict[run][ar]['t_total'])
				t_total_avg = statistics.mean(stats_dict[run][ar]['t_total'])
				
			t_clust_max = -1
			t_clust_avg = -1
			if 't__clustering' in stats_dict[run][ar].keys():
				t_clust_max = max(stats_dict[run][ar]['t__clustering'])
				t_clust_avg = statistics.mean(stats_dict[run][ar]['t__clustering'])

			cuts_clust_max = -1
			cuts_clust_avg = -1
			if 'cuts_clust' in stats_dict[run][ar].keys():
				cuts_clust_max = max(stats_dict[run][ar]['cuts_clust'])
				cuts_clust_avg = statistics.mean(stats_dict[run][ar]['cuts_clust'])

			cuts_extra_max = -1
			cuts_extra_avg = -1
			if 'cuts_extra' in stats_dict[run][ar].keys():
				cuts_extra_max = max(stats_dict[run][ar]['cuts_extra'])
				cuts_extra_avg = statistics.mean(stats_dict[run][ar]['cuts_extra'])
				
			s_info_out += '%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,%05.3f,' % \
												(worst_score,best_score,mean_score,
												 t_total_max, t_total_avg,
												 t_clust_max, t_clust_avg,
												 cuts_clust_max, cuts_clust_avg, 
												 cuts_extra_max, cuts_extra_avg)
						
		s_info_out += '%d' % missing_files[run]
		
		print(s_info_out)
		fp.write(s_info_out+'\n')			
			
else:

	s_info_out = ('%-30s %6s, %6s, %6s, %6s, %6s, %6s' % ('Method',
				  'Worst','Best','Mean',
				  'Worst','Best','Mean',))
	print(s_info_out)


	for run in evals.keys():
		run_name = run.replace('_',',')
		s_info_out = '%-30s ' % (run_name.replace('_',','))
		for ar in evals[run].keys():
			users_ious = []
			for user in range(6):
				users_ious.append(statistics.mean(evals[run][ar][user]))
			best_score = max(users_ious)*100
			worst_score = min(users_ious)*100
			mean_score = statistics.mean(users_ious)*100

				
			s_info_out += '%6.1f, %6.1f, %6.1f, ' % \
												(worst_score,best_score,mean_score)
						

		
		print(s_info_out)
	
