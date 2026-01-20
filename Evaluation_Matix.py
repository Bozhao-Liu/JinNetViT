import os
from typing import Union
import torch
import torch.nn as nn
import logging
from sklearn import metrics
import numpy as np
#import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

epslon = 1e-8

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
	def __call__(self):
		return self.avg
		
class EvalMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, size:int = 12):
		self.size = size
		self.reset()
		

	def reset(self):
		self.tp = np.zeros(self.size)
		self.fp = np.zeros(self.size)
		self.tn = np.zeros(self.size)
		self.fn = np.zeros(self.size)

	def update(self, pred, label):
		assert pred.shape[1] == self.size, 'Expected prediction dimension of {}, but got {}'.format(self.size, pred.shape[1])
		
		assert label.shape[1] == self.size, 'Expected label dimension of {}, but got {}'.format(self.size, label.shape[1])

		self.tp = self.tp + np.sum(label * np.array(pred==label).astype(float), axis = (0,2,3))
		self.fp = self.fp + np.sum((1-label) * np.array(pred!=label).astype(float), axis = (0,2,3))
		self.tn = self.tn + np.sum((1-label) * np.array(pred==label).astype(float), axis = (0,2,3))
		self.fn = self.fn + np.sum(label * np.array(pred!=label).astype(float), axis = (0,2,3))

	def precision(self):
		return self.tp/(self.tp+ self.fp)

	def recall(self):
		return self.tp/(self.tp+ self.fn)

	def F1(self):
		return 2*self.precision()*self.recall()/(self.precision()+self.recall())
		
	def acc(self):
		return (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
		
	def dict(self):
		return {'precision':self.precision(), 'recall':self.recall(), 'F1':self.F1(), 'acc': self.acc()}
		
class DiceLoss(nn.Module):
	def __init__(self, smooth=1e-6):
		"""
		Dice Loss for segmentation tasks.

		Args:
			smooth (float): Smoothing constant to avoid division by zero.
		"""
		super(DiceLoss, self).__init__()
		self.smooth = smooth

	def forward(self, inputs, targets):
		"""
		Compute Dice loss.

		Args:
			inputs (Tensor): Predicted logits or probabilities (N, C, H, W).
			targets (Tensor): Ground truth binary mask (N, C, H, W).

		Returns:
			Tensor: Dice loss (scalar if reduction != 'none').
		"""
		# Apply sigmoid if inputs are raw logits (for binary segmentation)
		inputs = torch.sigmoid(inputs)

		# Flatten per sample
		inputs = inputs.contiguous().view(inputs.shape[0], -1)
		targets = targets.contiguous().view(targets.shape[0], -1)

		intersection = (inputs * targets).sum(dim=1)
		union = inputs.sum(dim=1) + targets.sum(dim=1)

		dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
		return 1 - dice_score.mean()
		
		
class ThresholdMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, num_classes:int = 12):
		self.num_classes = num_classes
		self.thresholds = np.array(range(40,77,2))/100
		self.reset()

	def reset(self):
		self.union = defaultdict(lambda: 0)
		self.intersection = defaultdict(lambda: 0)

	def update(self, pred, label):
		assert pred.shape[1] == self.num_classes, 'Expected prediction dimension of {}, but got {}'.format(self.num_classes, pred.shape[1])
		
		assert label.shape[1] == self.num_classes, 'Expected label dimension of {}, but got {}'.format(self.num_classes, label.shape[1])
		label = np.asarray(label).astype(bool)

		for threshold in self.thresholds:
			predict = np.asarray(pred)>threshold
			inter = np.sum(np.logical_and(predict, label))
			uni = np.sum(np.logical_or(predict, label))
			self.union[threshold] += uni
			self.intersection[threshold] += inter

	def __call__(self):
		IOU = 0
		IOUs = {}
		threshold = 0
		for key in self.union:
			IOUs[key] = self.intersection[key] /self.union[key]
			if self.intersection[key] /self.union[key]>IOU:
				IOU = self.intersection[key] /self.union[key]
				threshold = key
				
		return [threshold], IOUs
		
class RecallLoss(nn.Module):
	""" An unofficial implementation of
		<Recall Loss for Imbalanced Image Classification and Semantic Segmentation>
		Created by: Zhang Shuai
		Email: shuaizzz666@gmail.com
		recall = TP / (TP + FN)
	Args:
		weight: An array of shape [C,]
		predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
		target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
	Return:
		diceloss
	"""
	def __init__(self, weight=None):
		super(RecallLoss, self).__init__()
		if weight is not None:
			weight = torch.Tensor(weight)
			self.weight = weight / torch.sum(weight) # Normalized weight
		self.smooth = 1e-5

	def forward(self, input, target):
		N, C = input.size()[:2]
		_, ind = torch.max(input, 1)# # (N, C, ) ==> (N, 1,)
		predict = torch.zeros(input.size()).cuda()

		for i in range(len(ind)):
			predict[i][ind[i]]=1


		true_positive = torch.sum(predict * target, dim=0)
		positive = torch.sum(target, dim=0)

		recall = (true_positive + self.smooth) / (positive + self.smooth)  # (N, C)

		if hasattr(self, 'weight'):
			if self.weight.type() != input.type():
				self.weight = self.weight.type_as(input)
				recall = recall * self.weight * C  # (N, C)
		recall_loss = 1 - recall  # 1
		pos_loss = -torch.mul(recall_loss, torch.log(input))
		pos_loss = torch.mul(target, pos_loss)
		neg_loss = -torch.mul(1-recall_loss, torch.log(1-input))
		neg_loss = torch.mul(1-target, neg_loss)

		recall_loss = pos_loss + neg_loss

		return torch.mean(recall_loss)
		

class RangeLoss(nn.Module):
	"""
		Range_loss = alpha * intra_class_loss + beta * inter_class_loss
		intra_class_loss is the harmonic mean value of the top_k largest distances beturn intra_class_pairs
		inter_class_loss is the shortest distance between different class centers
	"""
	def __init__(self, k=2, margin=0.1, alpha=0.5, beta=0.5, use_gpu=True, ordered=True, ids_per_batch=100, imgs_per_id=1):
		super(RangeLoss, self).__init__()
		self.use_gpu = use_gpu
		self.margin = margin
		self.k = k
		self.alpha = alpha
		self.beta = beta
		self.ordered = ordered
		self.ids_per_batch = ids_per_batch
		self.imgs_per_id = imgs_per_id

	def _pairwise_distance(self, features):
		"""
		 Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
		 Return: 
			pairwise distance matrix with shape(batch_size, batch_size)
		"""
		n = features.size(0)
		dist = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(n, n)
		dist = dist + dist.t()
		dist.addmm_(1, -2, features, features.t())
		dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
		return dist

	def _compute_top_k(self, features):
		"""
		 Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
		 Return: 
			top_k largest distances
		"""
	
		dist_array = self._pairwise_distance(features)
		dist_array = dist_array.view(1, -1)
		top_k = dist_array.sort()[0][0, -self.k * 2::2]	 # Because there are 2 same value of same feature pair in the dist_array
		# print('top k intra class dist:', top_k)
		
		# reading the codes below can help understand better
		'''
		dist_array_2 = self._pairwise_distance(features)
		n = features.size(0)
		mask = torch.zeros(n, n)
		if self.use_gpu: mask=mask.cuda()
		for i in range(0, n):
			for j in range(i+1, n):
				mask[i, j] += 1
		dist_array_2 = dist_array_2 * mask
		dist_array_2 = dist_array_2.view(1, -1)
		dist_array_2 = dist_array_2[torch.gt(dist_array_2, 0)]
		top_k_2 = dist_array_2.sort()[0][-self.k:]
		print(top_k_2)
		'''
		return top_k

	def _compute_min_dist(self, center_features):
		"""
		 Args:
			center_features: center matrix (before softmax) with shape (center_number, center_dim)
		 Return: 
			minimum center distance
		"""
		'''
		# reading codes below can help understand better
		dist_array = self._pairwise_distance(center_features)
		n = center_features.size(0)
		mask = torch.zeros(n, n)
		if self.use_gpu: mask=mask.cuda()
		for i in range(0, n):
			for j in range(i + 1, n):
				mask[i, j] += 1
		dist_array *= mask
		dist_array = dist_array.view(1, -1)
		dist_array = dist_array[torch.gt(dist_array, 0)]
		min_inter_class_dist = dist_array.min()
		print(min_inter_class_dist)
		'''
		n = center_features.size(0)
		dist_array2 = self._pairwise_distance(center_features)
		min_inter_class_dist2 = dist_array2.view(1, -1).sort()[0][0][n]  # exclude self compare, the first one is the min_inter_class_dist
		return min_inter_class_dist2

	def _calculate_centers(self, features, targets, ordered=True, ids_per_batch=32, imgs_per_id=4):
		"""
		 Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
			targets: ground truth labels with shape (batch_size)
			ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
			ids_per_batch: num of different ids per batch
			imgs_per_id: num of images per id
		 Return: 
			center_features: center matrix (before softmax) with shape (center_number, center_dim)
		"""
		if self.use_gpu:
			if ordered:
				if targets.size(0) == ids_per_batch * imgs_per_id:
					unique_labels = targets[0:targets.size(0):imgs_per_id]
				else:
					unique_labels = targets.cpu().unique().cuda()
			else:
				unique_labels = targets.cpu().unique().cuda()
		else:
			if ordered:
				if targets.size(0) == ids_per_batch * imgs_per_id:
					unique_labels = targets[0:targets.size(0):imgs_per_id]
				else:
					unique_labels = targets.unique()
			else:
				unique_labels = targets.unique()
		center_features = torch.zeros(unique_labels.size(0), features.size(1))
		if self.use_gpu:
			center_features = center_features.cuda()

		for i in range(unique_labels.size(0)):
			label = unique_labels[i]
			same_class_features = features[targets == label]
			center_features[i] = same_class_features.mean(dim=0)
		return center_features

	def _inter_class_loss(self, features, targets, ordered=True, ids_per_batch=32, imgs_per_id=4):
		"""
		 Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
			targets: ground truth labels with shape (batch_size)
			margin: inter class ringe loss margin
			ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
			ids_per_batch: num of different ids per batch
			imgs_per_id: num of images per id
		 Return: 
			inter_class_loss
		"""
		center_features = self._calculate_centers(features, targets, ordered, ids_per_batch, imgs_per_id)
		min_inter_class_center_distance = self._compute_min_dist(center_features)
		# print('min_inter_class_center_dist:', min_inter_class_center_distance)
		return torch.relu(self.margin - min_inter_class_center_distance)

	def _intra_class_loss(self, features, targets, ordered=True, ids_per_batch=32, imgs_per_id=4):
		"""
		 Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
			targets: ground truth labels with shape (batch_size)
			ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
			ids_per_batch: num of different ids per batch
			imgs_per_id: num of images per id
		 Return: 
			intra_class_loss
		"""
		if self.use_gpu:
			if ordered:
				if targets.size(0) == ids_per_batch * imgs_per_id:
					unique_labels = targets[0:targets.size(0):imgs_per_id]
				else:
					unique_labels = targets.cpu().unique().cuda()
			else:
				unique_labels = targets.cpu().unique().cuda()
		else:
			if ordered:
				if targets.size(0) == ids_per_batch * imgs_per_id:
					unique_labels = targets[0:targets.size(0):imgs_per_id]
				else:
					unique_labels = targets.unique()
			else:
				unique_labels = targets.unique()

		intra_distance = torch.zeros(unique_labels.size(0))
		if self.use_gpu:
			intra_distance = intra_distance.cuda()

		for i in range(unique_labels.size(0)):
			label = unique_labels[i]
			same_class_distances = 1.0 / self._compute_top_k(features[targets == label])
			intra_distance[i] = self.k / torch.sum(same_class_distances)
		# print('intra_distace:', intra_distance)
		return torch.sum(intra_distance)

	def _range_loss(self, features, targets, ordered=True, ids_per_batch=32, imgs_per_id=4):
		"""
		Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
			targets: ground truth labels with shape (batch_size)
			ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
			ids_per_batch: num of different ids per batch
			imgs_per_id: num of images per id
		Return:
			 range_loss
		"""
		inter_class_loss = self._inter_class_loss(features, targets, ordered, ids_per_batch, imgs_per_id)
		intra_class_loss = self._intra_class_loss(features, targets, ordered, ids_per_batch, imgs_per_id)
		range_loss = self.alpha * intra_class_loss + self.beta * inter_class_loss
		return inter_class_loss, intra_class_loss, range_loss

	def forward(self, features, targets):
		"""
		Args:
			features: prediction matrix (before softmax) with shape (batch_size, feature_dim)
			targets: ground truth labels with shape (batch_size)
			ordered: bool type. If the train data per batch are formed as p*k, where p is the num of ids per batch and k is the num of images per id.
			ids_per_batch: num of different ids per batch
			imgs_per_id: num of images per id
		Return:
			 range_loss
		"""
		if len(targets.size()) > 1:
			targets = torch.argmax(targets, dim = 1)
		assert features.size(0) == targets.size(0), "features.size(0) is not equal to targets.size(0)"
		if self.use_gpu:
			features = features.cuda()
			targets = targets.cuda()

		inter_class_loss, intra_class_loss, range_loss = self._range_loss(features, targets, self.ordered, self.ids_per_batch, self.imgs_per_id)
		return range_loss


class ECE_loss(nn.Module):
	def __init__(self, weights: Union[tuple, int] = (1, 1)):
		super(ECE_loss, self).__init__()
		if type(weights) is int:
			self.weight = [weights, weights]
		else:
			self.weight = [weights[0], weights[1]]

	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		pos_loss = torch.mul(label, 1.0/(pred + epslon) - 1)
		neg_loss = torch.mul(1 - label, 1.0/(1 - pred + epslon) - 1)
		loss =  self.weight[0]*pos_loss + self.weight[1]*neg_loss
		return torch.mean(loss)
		
		
class Focal_loss(nn.Module):
	def __init__(self, gamma: Union[tuple, int] =  (2, 2)):
		super(Focal_loss, self).__init__()
		if type(gamma) is int:
			self.gamma = [gamma, gamma]
		else:
			self.gamma = [gamma[0], gamma[1]]
			
	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		exp_pos = torch.mul(label, -torch.pow(1-pred, self.gamma[0]))
		pos_loss = torch.mul(exp_pos, torch.log(pred))
		exp_neg = torch.mul(1 - label, -torch.pow(pred, self.gamma[0]))
		neg_loss = torch.mul(exp_neg, torch.log(1-pred))
		loss =  pos_loss + neg_loss
		loss = torch.mean(loss) 
		if not np.isnan(loss.cpu().data.numpy().any()):	
			return loss
		else:
			print('#####NAN#####')
			return 1e5
		
class F_ECE_loss(nn.Module):
	def __init__(self, gamma: int = 2):
		super(F_ECE_loss, self).__init__()
		self.gamma = gamma

	def forward(self, pred, label):
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		pos_loss = torch.mul(label, 1.0/(pred + epslon) - 1)
		
		exp_neg = torch.mul(1 - label, -torch.pow(pred, self.gamma))
		neg_loss = torch.mul(exp_neg, torch.log(1-pred))
		loss =  pos_loss + neg_loss
		return torch.mean(loss)

class ASL(nn.Module):
	''' Notice - optimized version, minimizes memory allocation and gpu uploading,
	favors inplace operations'''

	def __init__(self, gamma_neg=4, gamma_pos=1):
		super(ASL, self).__init__()

		self.gamma_neg = gamma_neg
		self.gamma_pos = gamma_pos

	def forward(self, pred, label):
		""""
		Parameters
		----------
		x: input logits
		y: targets (multi-label binarized vector)
		"""
		pred = torch.clamp(pred, min=epslon, max=1-epslon)
		neg_label = 1 - label #1-y

		# Calculating Probabilities
		P_pos = pred
		P_neg = 1.0 - pred

		P_pos.clamp_(min=epslon)
		P_neg.add_(0.05).clamp_(max=1, min=epslon)

		# Basic CE calculation
		loss_pos = -torch.mul(label, torch.log(P_pos))
		loss_neg = -torch.mul(neg_label, torch.log(P_neg))

		# Asymmetric Focusing
		if self.gamma_neg > 0 or self.gamma_pos > 0:
			loss_pos = torch.mul(torch.pow(P_neg, self.gamma_pos), loss_pos)
			loss_neg = torch.mul(torch.pow(P_pos, self.gamma_neg), loss_neg)
						  
		loss = torch.add(loss_pos, loss_neg)
				
		return loss.mean()


def get_loss(loss_name, Hyperparam):
	loss_name = loss_name.lower()
	loss_name = loss_name.replace('\r', '')
	loss_name = loss_name.replace(' ', '')
	if loss_name == 'bce':
		return nn.BCEWithLogitsLoss()
	elif loss_name == 'ece': 
		return ECE_loss()
	elif loss_name == 'focal': 
		Hyperparam.gamma = 2
		return Focal_loss(Hyperparam.gamma)
	elif loss_name == 'f-ece': 
		Hyperparam.gamma = 2
		return F_ECE_loss(Hyperparam.gamma)
	elif loss_name == 'asl':
		return ASL()
	elif loss_name == 'recall': 
		return RecallLoss()
	elif loss_name == 'LDAM': 
		Hyperparam.gamma = 2
		return F_ECE_loss(Hyperparam.gamma)
	elif loss_name == 'MM':
		return ASL()
	elif loss_name == 'rangeloss':
		return RangeLoss(ids_per_batch=100, imgs_per_id=1)
	elif loss_name == 'costsensitiveloss':
		return CostSensitiveLoss(n_classes = 10)
	else:
		logging.error("No loss function with the name {} found, please check your spelling.".format(loss_name))
		logging.error("loss function List:")
		logging.error("	BCE")
		logging.error("	ECE")
		logging.error("	focal")
		logging.error("	ASL")
		logging.error("	F-ECE")
		logging.error("	Recall")
		logging.error("	LDAM")
		logging.error("	MM")
		logging.error("	RangeLoss")
		logging.error("	CostSensitiveLoss")
		import sys
		sys.exit()
		
		
def get_AUC(outputs):
	AUC = []
	for i in range(outputs[0].shape[1]):
		fpr, tpr, thresholds = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		AUC.append(metrics.auc(fpr, tpr))
	return np.mean(AUC)
	
def get_threshold(outputs, network:str, load_threshold:bool = True, CViter:tuple = (0, 1)):
	def get_dirname():
		path = './Result'
		if not os.path.exists(path):
			os.makedirs(path)
			
		path = os.path.join(path, 'Threshold')
		if not os.path.exists(path):
			os.makedirs(path)
			
		path = os.path.join(path, network)	
		if not os.path.exists(path):
			os.makedirs(path)
			
		cv_iter = '_'.join(tuple(map(str, CViter)))
		path = os.path.join(path, cv_iter)	
		if not os.path.exists(path):
			os.makedirs(path)
			load_threshold = False
			
		for i in range(outputs[0].shape[1]):
			subpath = os.path.join(path, str(i))
			if not os.path.exists(subpath):
				load_threshold = False
			
		return path
		
	def save_threshold(path, t):
		t =  np.array(t)
		for i in range(t.shape[0]):
			np.savetxt(os.path.join(path, str(i)), t)
		
	def load_threshold(path):
		t = []
		for i in range(output[0].shape[1]):
			t.append(np.loadtxt(os.path.join(path, str(i))))
			
		return t
	
	t_path = get_dirname()
	
	if load_threshold:
		return load_threshold(t_path)
		
	thresholds = []
	for i in range(output[0].shape[1]):
		threshold = []
		for x in range(output[0].shape[2]):
			thresholdx = []
			for y in range(output[0].shape[3]):
				precision, recall, t = metrics.precision_recall_curve(output[1][:, i, x, y], output[0][:, i, x, y], pos_label = 1 )
				f1_scores = 2*recall*precision/(recall+precision + 1e-8)
				ind = np.argmax(f1_scores)
				t = t[ind]
				thresholdx.append(t)
				#print(t)
			threshold.append(thresholdx)
		thresholds.append(threshold)
		
	save_threshold(t_path, thresholds)
		
	return thresholds
	
def get_eval_multi(outputs, thresholds: list = [0.5]*12):
	result = defaultdict(list)

	for i in range(outputs[0].shape[1]):
		fpr, tpr, _ = metrics.roc_curve(outputs[1][:, i], outputs[0][:, i], pos_label=1)
		result['AUC'].append(metrics.auc(fpr, tpr))		
		outputs[0][:, i] = outputs[0][:, i] > thresholds[i]
		
		result['threshold'].append(thresholds[i])
		result['acc'].append(metrics.accuracy_score(outputs[1][:, i], outputs[0][:, i]))
		result['Precision'].append(metrics.precision_score(outputs[1][:, i], outputs[0][:, i], average='weighted', zero_division = 0))
		result['Recall'].append(metrics.recall_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
		result['F0.5'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0.5))
		result['F0'].append(metrics.fbeta_score(outputs[1][:, i], outputs[0][:, i], average='weighted', beta=0))
		result['F1'].append(metrics.f1_score(outputs[1][:, i], outputs[0][:, i], average='weighted'))
	
	return result
	
'''
def plot_AUC_SD(netlist, evalmatices):
	plt.clf()
	possitive_ratio = np.loadtxt("./data/possitive_ratio.txt", dtype=float)
	logging.warning('	Creating standard diviation image for {}'.format('-'.join(netlist)))
	png_file = 'Crossvalidation_Analysis_{}.tex'.format('-'.join(netlist))

	if len(netlist) == 0:
		return


	plt.clf()
	fig, ax = plt.subplots(2)
	fig.suptitle('Accruacy, F1 for {}'.format('-'.join(netlist)))
	
	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[0])

	ax[0].boxplot(data, showfliers=False)
	ax[0].set_ylabel('Accruacy')

	data = []
	for net in netlist:
		data.append(np.array(evalmatices[net]).T[1])

	ax[1].boxplot(data, showfliers=False)
	ax[1].set_ylabel('F1')
	ax[1].set_xticklabels(netlist, fontsize=10)
	
	import tikzplotlib
	
	logging.warning('	Saving standard diviation image for {} \n'.format('-'.join(netlist)))
	tikzplotlib.save(png_file)
'''
	
