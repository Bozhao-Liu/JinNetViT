import os
import shutil
from Evaluation.getmatrix import save_matric_from_path
from Evaluation.create_stat import generate_pred_masks, generate_stat_file
from tqdm import tqdm
import numpy as np
import argparse

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
		  
parser = argparse.ArgumentParser(description='batch wise result analysis')
parser.add_argument('--load', default = True, type=str2bool, 
			help="specify whether load result from previously acquired matrics (default: True)")

def create_table(path = '.'):
	args,unknown = parser.parse_known_args()
	title = 'net & loss'
	table = ' \\\\ \n'
	
	keys = ["best-threshold", "IoU", "miss", "Dice-Coeff", "BIoU", "HD95", "MSD"]
	for key in keys:
		title = title + ' & ' + key
	path = os.path.join(path, 'Result', 'prediction')
	networks = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and folder != '__pycache__']
	priority_order = {
						"unet": 0,                 # 2015
						"deeplabv3": 1,            # 2017
						"mobilenetv2_unet": 2,     # 2018
						"mobilenetv3_unet": 3,     # 2019
						"transunet": 4,            # 2021
						"unett": 5,                # 2021 (UNet Transformer)
						"segformer": 6,            # 2021
						"swinunet": 7,             # 2021
						"medt": 8,                 # 2021
						"missformer": 9,           # 2023
						"medformer": 10,           # 2024
						"nnunet": 11,              # 2024
						"lightawnet": 12,          # 2025
						"msegnet": 13,             # 2025

						# Your models placed at the end
						"jin": 14,
						"jinpp": 15,
						"jinppvit": 16,
						"ujin": 17
					}
	networks = sorted(networks, key=lambda item: priority_order.get(item, float('inf')))
	with tqdm(total = len(networks), desc = 'networks') as t:
		for network in networks:
			t.set_description("network = {}".format(network))
			line = network
			network = os.path.join(path, network)
			losses = [folder for folder in os.listdir(network) if os.path.isdir(os.path.join(network, folder)) and folder != '__pycache__']
			for loss in tqdm(losses, desc = 'loss', leave = 0):
				result = line + ' & ' + loss
				loss = os.path.join(network, loss)
				shutil.copyfile(os.path.join('Evaluation', 'getmatrix.py'), os.path.join(loss, 'getmatrix.py'))	
				shutil.copyfile(os.path.join('Evaluation', 'create_stat.py'), os.path.join(loss, 'create_stat.py'))	
				metric = save_matric_from_path(loss, read_from_history = args.load)
				generate_pred_masks(loss, read_from_history = args.load)
				generate_stat_file(loss, read_from_history = args.load)
				for key in keys:
					if key in ["HD95", "MSD"]:
						result = result + ' & ' + '{}±{} '.format(np.round(np.mean(metric[key]), decimals=1), np.round(np.std(metric[key]), decimals=1))
					else:
						result = result + ' & ' + '{}±{} '.format(np.round(np.mean(metric[key])*100, decimals=1), np.round(np.std(metric[key])*100, decimals=1))
					
				for key in metric:
					if key not in keys:
						keys.append(key)
						title = title + ' & ' + key
						result = result + ' & ' + '{}±{} '.format(np.round(np.mean(metric[key])*100, decimals=1), np.round(np.std(metric[key])*100, decimals=1))
						
				with open(os.path.join(loss, 'table.txt'), 'w') as f:
					f.write(result)
					
				table = table + result + ' \\\\ \n'	
			t.update()

	with open(os.path.join(path, 'table.txt'), 'w') as f:
				f.write(title + table)
			

if __name__ == '__main__':
	create_table('.')
	import Evaluation.ttest
	import Evaluation.WSR
