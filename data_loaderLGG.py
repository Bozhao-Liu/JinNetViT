import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import Tensor
from itertools import chain
import torchvision.transforms as transforms
import pandas as pd
from torchvision.datasets import MNIST
from torchvision.io import read_image 
from PIL import Image
from tqdm import tqdm

np.random.seed(230)
class DatasetWrapper(object):

	class __DatasetWrapper(object):
		"""
		A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
		"""

		def __init__(self, params):	
			assert params.CV_iters > 2, 'Cross validation folds must be more than 2 folds'
			self.cv_iters = params.CV_iters
			
			datapath = '../../data/LGG'
			self.dataset = []
			folders = os.listdir(datapath)
			folders.sort()
			for folder in tqdm(folders):
				folder = os.path.join(datapath, folder)
				if os.path.isdir(folder):
					imgs = os.listdir(folder)
					imgs.sort()
					self.dataset = self.dataset + [os.path.join(folder, image) for image in imgs if '_box.png' in image]

			self.dataset.sort()
					
			self.ind = np.arange(len(self.dataset))
			self.ind = self.ind[:int(len(self.ind)/self.cv_iters) * self.cv_iters]
			self.ind = self.ind.reshape((self.cv_iters, -1))

	instance = None
	def __init__(self, params, CViters):
		super(DatasetWrapper, self).__init__()
			
		if not DatasetWrapper.instance:
			DatasetWrapper.instance =  DatasetWrapper.__DatasetWrapper(params)
		self.cv_iters = params.CV_iters
		self.num_classes = params.channels
		DatasetWrapper.Testindex = CViters[0]
		DatasetWrapper.CVindex = CViters[1]

	'''	
	def __getattr__(self, name):
		return getattr(DatasetWrapper.instance, name)
	'''
	
	def features(self, key):
		"""
		Args: 
			key:(string) value from dataset	
		Returns:
			features in list	
		"""
		return DatasetWrapper.instance.dataset[key].replace("_box.png", ".png")

	def label(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:
			arrayed label
		"""
		return DatasetWrapper.instance.dataset[key].replace("box.png", "mask.png")
		
	def box(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:

			arrayed label
		"""
		return DatasetWrapper.instance.dataset[key]


	def __trainSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of trainning set
		"""

		ind = list(range(self.cv_iters))
		ind = np.delete(ind, [self.CVindex, self.Testindex])

		trainSet = DatasetWrapper.instance.ind[ind].flatten()
		np.random.shuffle(trainSet)
		return trainSet
	
	def __valSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of validation set
		"""

		valSet = DatasetWrapper.instance.ind[self.CVindex].flatten()
		np.random.shuffle(valSet)
		return valSet

	def __testSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of full dataset
		"""

		testSet = DatasetWrapper.instance.ind[self.Testindex].flatten()
		return testSet

	def getDataSet(self, dataSetType = 'train'):
		"""
		Args: 
			dataSetType: (string) 'train' or 'val'	
		Returns:
			dataset: (np.ndarray) array of key/id of data set
		"""

		if dataSetType == 'train':
			return self.__trainSet()

		if dataSetType == 'val':
			return self.__valSet()

		if dataSetType == 'test':
			return self.__testSet()

		return self.__testSet()
		
class Normalize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor: Tensor) -> Tensor:
        return (tensor - torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))

class imageDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, dataSetType, params, CViters):
		"""
		initialize DatasetWrapper
		"""
		super(imageDataset, self).__init__()
		self.DatasetWrapper = DatasetWrapper(params, CViters)
		self.dataSetType = dataSetType
		self.samples = self.DatasetWrapper.getDataSet(dataSetType)
				
		self.transformer = transforms.ToTensor()



	def __len__(self):
		# return size of dataset
		return len(self.samples)



	def __getitem__(self, idx):
		"""
		Fetch feature and labels from dataset using index of the sample.

		Args:
		    idx: (int) index of the sample

		Returns:
		    feature: (Tensor) feature array
		    label: (Tensor) corresponding label of sample
		"""
		sample = self.samples[idx]
		data = Image.open(self.DatasetWrapper.features(sample))
		image = self.transformer(data)
		
		label= Image.open(self.DatasetWrapper.label(sample)).convert("L")
		
		label = self.transformer(label).to(int)
		assert (label >= 0).all() and (label <= 1).all(), '{self.dataSetType} image segmentation mask out of range' + self.DatasetWrapper.label(sample)

		return image, label

def fetch_dataloader(types, params, CViters):
	"""
	Fetches the DataLoader object for each type in types.

	Args:
	types: (list) has one or more of 'train', 'val'depending on which data is required '' to get the full dataSet
	params: (Params) hyperparameters

	Returns:
	data: (dict) contains the DataLoader object for each type in types
	"""
	dataloaders = {}
	assert CViters[0] != CViters[1], 'ERROR! Test set and validation set cannot be the same!'
	
	if len(types)>0:
		for split in types:
			if split in ['train', 'val']:
				dl = DataLoader(imageDataset(split, params, CViters), 
						batch_size=params.batch_size, 
						shuffle=True,
						num_workers=params.num_workers,
						pin_memory=params.cuda)

				dataloaders[split] = dl
			else:
				dl = DataLoader(imageDataset(split, params, CViters), 
						batch_size=params.batch_size*2, 
						shuffle=False,
						num_workers=params.num_workers,
						pin_memory=params.cuda)

				dataloaders[split] = dl
	else:
		dl = DataLoader(imageDataset('', params, CViters), 
				batch_size=params.batch_size*2, 
				shuffle=False,
				num_workers=params.num_workers,
				pin_memory=params.cuda)

		return dl

	return dataloaders



