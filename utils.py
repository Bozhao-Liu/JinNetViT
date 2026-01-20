import json
import logging
import os
import shutil
import torch
import numpy as np
from itertools import product
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url

try:
	from configparser import ConfigParser
except ImportError:
	from ConfigParser import ConfigParser  # ver. < 3.0

class Params():
	"""Class that loads hyperparameters from a json file.

	Example:
	```
	params = Params(json_path)
	print(params.learning_rate)
	params.learning_rate = 0.5  # change the value of learning_rate in params
	```
	"""

	def __init__(self, model_dir, network, paramtype = 'params', loss_fn = '', checkpoint = False):
		assert paramtype in ['params','Hyperparams'], 'Param type {} not found'.format(paramtype)

		self.Best_loss = np.inf
		json_path = os.path.join(model_dir, network)
		loss_fn = loss_fn.replace(" ", "")
		self.__json_file = os.path.join(json_path, '{}{}.json'.format(paramtype, loss_fn))
		if not os.path.isfile(self.__json_file):
			logging.warning("Can not find json file {}".format(self.__json_file))
			self.__json_file = os.path.join(json_path, '{}.json'.format(paramtype))
			
		#logging.warning("Loading json file {}".format(self.__json_file))
		assert os.path.isfile(self.__json_file), "Can not find File {}".format(self.__json_file)
		with open(self.__json_file) as f:
			params = json.load(f)

		self.__dict__.update(params)
		self.__json_file = os.path.join(json_path, '{}{}.json'.format(paramtype, loss_fn))
			
		if checkpoint and paramtype in ['Hyperparams']:
			self.__json_file = os.path.join(json_path, '{}_{}_checkpoint.json'.format(paramtype, loss_fn))
		
			logging.warning("Loading json file {}".format(self.__json_file))
				
			if os.path.isfile(self.__json_file):
				with open(self.__json_file) as f:
					params = json.load(f)
					self.__dict__.update(params)
			else:
				self.__dict__ = list(param_search_list(self))[0]

	def update(self, params):
		json_file = self.__json_file
		self.__dict__.update(params)
		self.__json_file = json_file
		
	def save(self):
		json_file = self.__json_file
		del self.__json_file
		with open(json_file, 'w') as f:
			json.dump(self.__dict__, f, indent=4)
		self.__json_file = json_file
		
	def reload(self):
		"""Loads parameters from json file"""
		with open(self.__json_file) as f:
			params = json.load(f)
			self.__dict__.update(params)

	def dict(self):
		"""Gives dict-like access to Params instance by `params.dict['learning_rate']"""
		return self.__dict__
        
def set_params(model_dir, network, paramtype):
	params = Params(model_dir, network, paramtype)

	# use GPU if available
	params.cuda = torch.cuda.is_available()

	# Set the random seed for reproducible experiments
	torch.manual_seed(230)
	if params.cuda: 
		torch.cuda.manual_seed(230)

	return params
	
def param_search_list(hyperparams):
	from collections import defaultdict
	params = defaultdict(list)
	for key in hyperparams.dict().keys():
		if key == 'learning_rate':
			params[key] = [1e-5, 5e-6, 1e-6]
		elif key == 'dropout_rate':
			params[key] = list(np.arange(0.1, 1.0, 0.2, dtype = float))	
		elif key == 'lrDecay':
			params[key] = list(np.arange(0.85, 0.96, 0.02, dtype = float))	
		elif 'layer' in key:
			params[key] = [96, 128, 192, 256]
		else:
			params[key] = [hyperparams.dict()[key]]
	keys = params.keys()
	vals = params.values()
	for instance in product(*vals):
		yield dict(zip(keys, instance))		
				
def set_logger(model_dir, network, level = 'info'):
	"""Set the logger to log info in terminal and file `log_path`.

	In general, it is useful to have a logger so that every output to the terminal is saved
	in a permanent file. Here we save it to `model_dir/train.log`.

	Example:
	```
	logging.info("Starting training...")
	```

	Args:
	log_path: (string) where to log
	"""
	log_path = os.path.join(model_dir, network)
	assert os.path.isdir(log_path), "Can not find Path {}".format(log_path)
	log_path = os.path.join(log_path, 'train.log')
	print('Saving {} log to {}'.format(level, log_path))
	level = level.lower()
	logger = logging.getLogger()
	if level == 'warning':
		level = logging.WARNING
	elif level == 'debug':
		level = logging.DEBUG
	elif level == 'error':
		level = logging.ERROR
	elif level == 'critical':
		level = logging.CRITICAL
	else:
		level = logging.INFO
	logger.setLevel(level)

	if not logger.handlers:
		# Logging to a file
		file_handler = logging.FileHandler(log_path)
		file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
		logger.addHandler(file_handler)

		# Logging to console
		stream_handler = logging.StreamHandler()
		stream_handler.setFormatter(logging.Formatter('%(message)s'))
		logger.addHandler(stream_handler)



def save_dict_to_json(d, json_path):
	"""Saves dict of floats in json file

	Args:
	d: (dict) of float-castable values (np.float, int, float, etc.)
	json_path: (string) path to json file
	"""
	with open(json_path, 'w') as f:
		# We need to convert the values to float for json (it doesn't accept np.array, np.float, )
		d = {k: float(v) for k, v in d.items()}
		json.dump(d, f, indent=4)

	
def get_checkpointname(args, checkpoint_type, CViter):
	checkpointpath = os.path.join(os.path.join(args.model_dir,'Model'), args.network)
	checkpointpath = os.path.join(checkpointpath, 'Checkpoints' + str(checkpoint_type))
	checkpointfile = os.path.join(checkpointpath, 
				'{network}_{cv_iter}_{gamma}.pth.tar'.format(network = args.network+args.loss, 
									     cv_iter = '_'.join(tuple(map(str, CViter))), gamma = args.gamma))
	return checkpointpath, checkpointfile									     

