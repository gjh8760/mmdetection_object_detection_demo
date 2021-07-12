import numpy as np
from copy import deepcopy
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_pickle', default='./prediction/93000장_focal_borderaug/publaynet/pred.pickle')
parser.add_argument('--gt_pickle', default='./GT_anns/93000장_focal_borderaug/publaynet/gt.pickle')
parser.add_argument('--conf_thr', default=0.7)
args = parser.parse_args()

pred_pickle = args.pred_pickle
gt_pickle = args.gt_pickle
conf_thr = args.conf_thr

# pred_pickle = './prediction/93000장_focal_borderaug/publaynet/pred.pickle'
# gt_pickle = './GT_anns/93000장_focal_borderaug/publaynet/gt.pickle'


def calc_iou(gt_bbox, pred_bbox):

	x1_gt, y1_gt, x2_gt, y2_gt = gt_bbox
	x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox

	area_gt = (x2_gt - x1_gt) * (y2_gt - y1_gt)
	area_pred = (x2_pred - x1_pred) * (y2_pred - y1_pred)

	x1_inter, y1_inter = max(x1_gt, x1_pred), max(y1_gt, y1_pred)
	w_inter, h_inter = max(0, min(x2_gt, x2_pred) - x1_inter), max(0, min(y2_gt, y2_pred) - y1_inter)
	area_inter = w_inter * h_inter
	area_union = area_gt + area_pred - area_inter
	
	return area_inter / area_union


def draw_bbox(img, bbox, color):
	result = deepcopy(img)
	x1, y1, x2, y2 = deepcopy(bbox)
	x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
	if color == 'red':
		cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
	else:
		cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
	return result


def get_single_image_results(gt_boxes_, pred_boxes_, conf_thr, iou_thr, mode='all'):
	"""
	gt_boxes: [array([x1, y1, x2, y2], [x1, y1, x2, y2], ...), array()]
	pred_boxes: [array([x1, y1, x2, y2, conf], [x1, y1, x2, y2, conf], ...), array()]
	
	Returns:
		dict: {'TP': int, 'FP': int, 'FN': int}
	"""
	assert mode in ['bordered', 'borderless', 'all']
	
	# gt_boxes, pred_boxes = deepcopy(gt_boxes_), deepcopy(pred_boxes_)

	if mode == 'bordered':
		gt_boxes, pred_boxes = gt_boxes_[0], pred_boxes_[0]
	elif mode == 'borderless':
		gt_boxes, pred_boxes = gt_boxes_[1], pred_boxes_[1]
	elif mode == 'all':
		gt_boxes = np.concatenate((gt_boxes_[0], gt_boxes_[1]), axis=0)
		pred_boxes = np.concatenate((pred_boxes_[0], pred_boxes_[1]), axis=0)

	pred_boxes_temp = []
	for pred_box in pred_boxes:
		conf = pred_box[-1]
		if conf >= conf_thr:
			pred_boxes_temp.append(pred_box)

	pred_boxes = pred_boxes_temp

	all_pred_indices = range(len(pred_boxes))
	all_gt_indices = range(len(gt_boxes))

	if len(all_pred_indices) == 0:
		tp = fp = 0
		fn = len(gt_boxes)
		return {'TP': tp, 'FP': fp, 'FN': fn}

	if len(all_gt_indices) == 0:
		tp = 0
		fp = len(pred_boxes)
		fn = 0
		return {'TP': tp, 'FP': fp, 'FN': fn}

	gt_idx_thr = []
	pred_idx_thr = []
	ious = []

	for ipb, pred_box in enumerate(pred_boxes):
		for igb, gt_box in enumerate(gt_boxes):
			iou = calc_iou(gt_box, pred_box[:-1])

			if iou >= iou_thr:
				gt_idx_thr.append(igb)
				pred_idx_thr.append(ipb)
				ious.append(iou)

	iou_sort = np.argsort(ious)[::1]

	if len(iou_sort) == 0:
		tp = 0
		fp = len(pred_boxes)
		fn = len(gt_boxes)
		return {'TP': tp, 'FP': fp, 'FN': fn}
	else:
		gt_match_idx = []
		pred_match_idx = []
		
		for idx in iou_sort:
			gt_idx = gt_idx_thr[idx]
			pred_idx = pred_idx_thr[idx]
		
			# If the boxes are unmatched, add them to matches
			if (gt_idx not in gt_match_idx) and (pred_idx not in pred_match_idx):
				gt_match_idx.append(gt_idx)
				pred_match_idx.append(pred_idx)

		tp = len(gt_match_idx)
		fp = len(pred_boxes) - len(pred_match_idx)
		fn = len(gt_boxes) - len(gt_match_idx)

	return {'TP': tp, 'FP': fp, 'FN': fn}


def get_images_results(gt_dict, pred_dict, conf_thr, iou_thr, mode='all'):
	
	tp = 0
	fp = 0
	fn = 0

	for filename in sorted(pred_dict.keys()):
		# if filename == 'PMC3006370_00004.jpg':
		# 	import ipdb
		# 	ipdb.set_trace()
		gt_boxes = gt_dict[filename]
		pred_boxes = pred_dict[filename]

		single_image_result = get_single_image_results(gt_boxes, pred_boxes, conf_thr, iou_thr, mode)
		# print(f'{filename}:', single_image_result)

		tp += single_image_result['TP']
		fp += single_image_result['FP']
		fn += single_image_result['FN']

	return {'TP': tp, 'FP': fp, 'FN': fn}


def calc_precision_recall_f1(images_result):
	tp = images_result['TP']
	fp = images_result['FP']
	fn = images_result['FN']

	try:
		precision = tp / (tp + fp)
	except ZeroDivisionError:
		precision = 0.

	try:
		recall = tp / (tp + fn)
	except ZeroDivisionError:
		recall = 0.

	try:
		f1 = 2 * precision * recall / (precision + recall)
	except ZeroDivisionError:
		f1 = 0.

	return precision, recall, f1


def main():

	"""
	pred_dict : dict.
		{'img1.png': [array, array], ...}
		array: [x1, y1, x2, y2, conf]
		bordered, borderless 순

	gt_dict: dict.
		{{'img1.png': [array, array], ...}}
		array: [x1, y1, x2, y2]
	"""

	with open(pred_pickle, 'rb') as f:
		pred_dict = pickle.load(f)
	with open(gt_pickle, 'rb') as f:
		gt_dict = pickle.load(f)

	ious = [0.6, 0.7, 0.8, 0.9]
	print('\n')
	for iou in ious:
		images_result = get_images_results(gt_dict, pred_dict, conf_thr=conf_thr, iou_thr=iou, mode='all')
		precision, recall, f1 = calc_precision_recall_f1(images_result)
		print('[iou=%f]' % iou)
		print(images_result)
		print(f'precision: {precision}, recall: {recall}, f1: {f1}')
		print('\n')


if __name__ == '__main__':
	main()