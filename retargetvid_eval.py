import os

	
with open('annots.pkl', 'rb') as f:
	annots = pickle.load(f)

with open('annots_names.pkl', 'rb') as f:
	annots_names = pickle.load(f)
	
runs_folder = 'results'
runs = [f.path for f in os.scandir(runs_folder) if f.is_dir()]
	
for run in runs:
	# eval
	print(' Evaluation::')
	if print_eval_frames:
		cap = cv2.VideoCapture(vid_path)
	for user in range(6):
		frames_ious=[]
		for iframe,method_bb in enumerate(bbs):
			# [i users][k mode][j video][l frame]
			annot_gt = annots[user][iorp][annots_inds.index(int(vid_fn))][iframe]
			if orp=='1:3':
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
			
			if print_eval_frames:
				print(annot_bb, '-', method_bb)
				flag, frame = cap.read()
				frame = cv2.rectangle(frame, (annot_bb[0], annot_bb[1]), (annot_bb[2], annot_bb[3]), (0,255,0), 1) # green gt
				frame = cv2.rectangle(frame, (method_bb[0], method_bb[1]), (method_bb[2], method_bb[3]), (255,0,0), 1) # blue result
				cv2.imshow('video', frame)
				cv2.waitKey(0)
		vid_iou = statistics_mean(frames_ious)
		evals[profile_name][orp][user].append(vid_iou)
		print(' user #%d: %.3f' % (user+1,vid_iou))
	
	# statistics
	with open('log.txt', 'a') as lfp:
		lfp.write('%-28s,%s,%.3f,%.1f,%.3f,%s\n' % \
			(vid_fn,
			orp,
			results_dict['coverage_score'],
			fc_test/fr_test,
			results_dict['percentage'],
			results_dict['results']))
		
	for k in sc_times_crop.keys():
		if 'int' in str(type(sc_times_crop[k])) or 'float' in str(type(sc_times_crop[k])):
			if k not in stats_dict.keys():
				stats_dict[k] = []
			stats_dict[k].append(float(sc_times_crop[k]))
		
	for k in results_dict.keys():
		if 'int' in str(type(results_dict[k])) or 'float' in str(type(results_dict[k])):
			if not '_t' in str(k):
				if k not in stats_dict.keys():
					stats_dict[k] = []
				stats_dict[k].append(float(results_dict[k]))
			
		
	print('\n Done processing video "%s" with "%s"\n' % (vid_fn, profile_name), flush=True)
	
print('Eval::')
for profile in evals.keys():
	for orp in evals[profile].keys():
		print('    - orp:', orp)
		print(' - profile:', profile)
		users_ious = []
		for user in range(6):
			users_ious.append(statistics_mean(evals[profile][orp][user]))
		best_score = max(users_ious)
		worst_score = min(users_ious)
		mean_score = statistics_mean(users_ious)
		print('         %.3f, %.3f, %.3f' % (worst_score*100,best_score*100,mean_score*100))
		print('')

		
for k in stats_dict.keys():
	try:
		print(' %-28s: [%7.3f,%7.3f],  avg:%7.3f,  med:%7.3f,  +/-:%7.3f)' % (str(k),
			min(stats_dict[k]),
			max(stats_dict[k]),
			statistics_mean(stats_dict[k]),
			np.median(stats_dict[k]),
			statistics_std(stats_dict[k])) )
	except:
		pass


for r in results_string:
	print(r)