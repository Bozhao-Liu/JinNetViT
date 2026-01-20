import argparse

import torch
import logging
import os

import numpy as np
from itertools import permutations, product
from Solver import Solver
import torch.backends.cudnn as cudnn
from model_loader import get_model_list
from utils import set_logger, set_params, Params
from Evaluation_Matix import get_eval_multi, get_threshold
from collections import defaultdict
import json
from tqdm import tqdm

CUDA_VISIBLE_DEVICES = 0

def str2bool(v):
    if isinstance(v, bool):
       return v
       
    v = v.replace('\r', '')
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--train', default = False, type=str2bool, 
			help="specify whether train the model or not (default: False)")
parser.add_argument('--model_dir', default='./', 
			help="Directory containing params.json")
parser.add_argument('--resume', default = True, type=str2bool, 
			help='resume from latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default = '',
			help='select network to train on. leave it blank means train on all model')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')
parser.add_argument('--gamma', default=2, type=float,
			help='gamma value for focal')	
parser.add_argument('--loss', type=str, default = 'BCE',
			help='select loss function to train with. ')
parser.add_argument('--dice', default = True, type=str2bool, 
			help="specify whether train with dice loss or not (default: True)")
parser.add_argument('--device', type=str, default = '0',
			help='select device to train with. ')
parser.add_argument('--batch_size', type=int, default = 0,
			help='batch_size to train with. ')

def main(args):
	assert torch.cuda.is_available(), "ERROR! GPU is not available."
	cudnn.benchmark = True
	
	print(args)
	print(torch.cuda.get_device_name())
	netlist = get_model_list(args.network)
	eval_matrix = {}
	for network in netlist:
		args.network = network
		set_logger(os.path.join(args.model_dir,'Model'), args.network, args.log)
		params = set_params(os.path.join(args.model_dir,'./Model'), network, paramtype = 'params')
		if args.batch_size:
			params.batch_size = args.batch_size
		print(params.batch_size)
		params.hyperparam = Params(model_dir = os.path.join(args.model_dir,'./Model'), network = network, paramtype = 'Hyperparams', loss_fn = args.loss)
		CV_iters = list(permutations(list(range(params.CV_iters)), 2))
		CV_iters = [(0, 4),(1, 4),(2, 4),(3, 4)]
		#logging.warning(params.hyperparam.dict())
		#CV_iters = [(0, 1)]
		eval_matrix = defaultdict(list)
		with tqdm(total = len(CV_iters)) as t:
			for i, CViter in enumerate(CV_iters):
				logging.info('Cross Validation on iteration {}/{CV_iters}, {network}, {loss}'.format(i+1, CV_iters = len(CV_iters), network = args.network, loss = args.loss))
				
				solver = Solver(args, params, CViter)
				
				if args.train:
					best_loss = solver.train()
					
				solver.test(solver.thresholds())
				
				
				t.update()
	from get_overleaf_table import create_table
	create_table('.')
	import Evaluation.ttest
	import Evaluation.WSR


		
if __name__ == '__main__':
	args,unknown = parser.parse_known_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.device
	main(args)
