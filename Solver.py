from numpy import isnan, inf, savetxt
import numpy as np
import torch
import logging, gc
from torch.utils.checkpoint import checkpoint
try:
	from torch.amp import GradScaler, autocast
except ImportError:
	from torch.cuda.amp import GradScaler, autocast

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from Evaluation_Matix import *
from utils import *
import model_loader
from data_loader import fetch_dataloader
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

class Solver:
	def __init__(self, args, params, CViter):
		def init_weights(m):
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				m.bias.data.fill_(0.01)
		torch.cuda.empty_cache() 
		self.args = args
		self.params = params
		self.CViter = CViter
		self.dataloaders = fetch_dataloader(['train', 'val', 'test'], params, CViter) 
		self.model = model_loader.loadModel(args.network, params.channels).cuda()
		#self.model.apply(init_weights)
		self.optimizer = torch.optim.Adam(	self.model.parameters(), 
							params.hyperparam.learning_rate, 
							betas=(0.9, 0.999), 
							eps=1e-08, 
							weight_decay = params.weight_decay, 
							amsgrad=False)
		self.loss_fn = get_loss(args.loss, params.hyperparam)
		self.diceloss = DiceLoss()
		self.scaler = GradScaler()
		
	def __step__(self):
		torch.cuda.empty_cache() 
		logging.info("Training")
		losses = AverageMeter()
		#acc = AverageMeter()
		# switch to train mode
		self.model.train()
		#with tqdm(total=len(self.dataloaders['train']), unit = 'epoch', leave = 0) as t:
		for i, (datas, label) in enumerate(self.dataloaders['train']):
			datas = datas.cuda().float()
			datas.requires_grad_(True)
			label = label.cuda().float()

			self.optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(device_type = 'cuda'):
				output = self.model(datas).float()
				
			cost = self.loss_fn(output, label.float())
			if self.args.dice:
				cost = cost + self.diceloss(output, label.float())

			# Scaled backward pass (AMP)
			self.scaler.scale(cost).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			losses.update(cost.detach().cpu().item(), len(datas))

			# Cleanup
			del output, cost
			gc.collect()

		return losses()
	
	
	def validate(self, dataset_type = 'val'):
		torch.cuda.empty_cache() 
		logging.info("Validating")
		losses = AverageMeter()
			
		# switch to evaluate mode
		self.model.eval()
		#with tqdm(total=len(self.dataloaders[dataset_type]), leave = 0) as t:
		for i, (datas, label) in (enumerate(self.dataloaders[dataset_type])):
			datas = datas.cuda()
			label = label.cuda()

			self.optimizer.zero_grad(set_to_none=True)

			with torch.amp.autocast(device_type = 'cuda'):
				output = self.model(datas).float()
			
			loss = self.loss_fn(output, label.float())
			if self.args.dice:
				loss = loss + self.diceloss(output, label.float())
			
			# measure record cost
			losses.update(loss.cpu().data.numpy(), len(datas))
			
			del output
			
			gc.collect()
			#t.set_postfix(loss = '{:05.3f}'.format(losses()))
			#t.update()
		

		return losses()
		
	def test(self, thr):
		from PIL import Image
		def saveimg(i, imgs, prop, pred, label, path):	
			imgpath = os.path.join(path, 'minibatch_' +  str(i))
			if not os.path.exists(imgpath):
				os.makedirs(imgpath)
			for m in range(imgs.size(0)):
				imgname = os.path.join(imgpath, 'img_' + str(m) + '.png')
				#print(imgs[img])
				save_image(torch.Tensor(imgs[m]), imgname)
				for p in range(pred.shape[1]):
					propname = os.path.join(imgpath, 'img_' + str(m) + '_condition_' + str(p) + 'prop.png')
					predname = os.path.join(imgpath, 'img_' + str(m) + '_condition_' + str(p) + 'pred.png')
					labelname = os.path.join(imgpath, 'img_' + str(m) + '_condition_' + str(p) + 'label_' + str(np.sum(label[m, p])>0) + '.png')
					
					img = Image.fromarray(np.uint8(pred[m, p] * 255) , 'L')
					img.save(predname)
					img = Image.fromarray(np.uint8(label[m, p] * 255) , 'L')
					img.save(labelname)
					img = Image.fromarray(np.uint8(prop[m, p] * 255) , 'L')
					img.save(propname)
					
		def createpath():
			import shutil
			path = './Result'
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, 'prediction')
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.network)	
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.loss)					
			if not os.path.exists(path):
				os.makedirs(path)
				
			if os.path.exists(os.path.join(path, 'matrix.json')):
				os.remove(os.path.join(path, 'matrix.json'))	
			
			cv_iter = '_'.join(tuple(map(str, self.CViter)))
			path = os.path.join(path, cv_iter)	
			if os.path.exists(path):
				import shutil
				shutil.rmtree(path, ignore_errors=True)
				
			os.makedirs(path)
			
			return path
			
		torch.cuda.empty_cache() 
		logging.info("testing")
		self.__resume_checkpoint__('best')
		
		savepath = createpath()
			
		matrics = EvalMeter(self.params.channels)
		# switch to evaluate mode
		self.model.eval()

		#with tqdm(total=len(self.dataloaders['test']), leave = 0) as t:
		for i, (datas, label) in (enumerate(self.dataloaders['test'])):
			logging.info("        Compute output")
			datas = datas.cuda()

			with torch.amp.autocast(device_type = 'cuda'):
				output = self.model(datas).detach().float()
				
			output = torch.sigmoid(output).cpu().numpy()
			label_var = label.detach().float().cpu().numpy()
			
			pred = output > np.repeat(thr, output.shape[2] * output.shape[3]).reshape(( output.shape[2], output.shape[3]))
			pred = pred.astype(int)
			saveimg(i, datas, output, pred, label_var, savepath)
			matrics.update(output, label_var)
			del output
			del label_var
			
			gc.collect()
				
				#t.set_postfix(gpu = torch.cuda.utilization(), refresh=True)
				#t.update()
				

		
		return matrics.dict()
		
	def thresholds(self, load:bool = True):
		def get_dirname(load):
			path = './Result'
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, 'Threshold')
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.network)	
			if not os.path.exists(path):
				os.makedirs(path)
				
			path = os.path.join(path, self.args.loss)	
			if not os.path.exists(path):
				os.makedirs(path)
				
			cv_iter = '_'.join(tuple(map(str, self.CViter)))
			path = os.path.join(path, cv_iter + '.txt')	
			if not os.path.exists(path):
				load = False
				
			return path, load
		
		def save_threshold(path, t):
			np.savetxt(path, np.array(t[0]))
			import json
			with open(path.replace("txt", "json"), 'w') as file:
				json.dump(t[1], file, indent=4)	
			
		def load_threshold(path):
			return np.loadtxt(path)
			
		t_path, load = get_dirname(load)	
		if load:
			return load_threshold(t_path)
			
		torch.cuda.empty_cache() 
		logging.info("testing")
		self.__resume_checkpoint__('best')
			
		meter = ThresholdMeter(self.params.channels)
		# switch to evaluate mode
		self.model.eval()
		#with tqdm(total=len(self.dataloaders['val']), leave = 0) as t:	
		for i, (datas, label) in (enumerate(self.dataloaders['val'])):
			logging.info("        Compute output")
			datas = datas.cuda()
			with torch.amp.autocast(device_type = 'cuda'):
				output = self.model(datas).detach().float()

			output = torch.sigmoid(output).cpu().numpy()
			label_var = label.float().cpu().numpy()
			logging.info("        meter.update")
			meter.update(output, label_var)
			
			del output
			del label_var
			
			gc.collect()

				#t.set_postfix(gpu = torch.cuda.utilization(), refresh=True)
				#t.update()
				
		save_threshold(t_path, meter())
		
		return meter()[0]


	def train(self):
		start_epoch = 0
		best_loss = inf	
		
		if self.args.resume:
			logging.info('Resuming Checkpoint')
			start_epoch, best_loss = self.__resume_checkpoint__('')
			if not start_epoch < self.params.epochs:
				logging.info('Skipping training for finished model\n')
				return 0
							
		scheduler = StepLR(self.optimizer, step_size=5, gamma=self.params.hyperparam.lrDecay)	
		logging.info('    Starting With Best loss = {loss:.4f}'.format(loss = best_loss))
		logging.info('Initialize training from {} to {} epochs'.format(start_epoch, self.params.epochs))
		with tqdm(total=self.params.epochs - start_epoch, leave = 0) as t:
			for epoch in range(start_epoch, self.params.epochs):
				logging.info('CV [{}], Training Epoch: [{}/{}]'.format('_'.join(tuple(map(str, self.CViter))), epoch+1, self.params.epochs))
				
				
				self.__step__()
				gc.collect()
				# evaluate on validation set
				loss = self.validate()
				
				gc.collect()

				# remember best model and save checkpoint
				logging.info('    loss {loss:.4f};\n'.format(loss = loss))		
				if loss < best_loss:
					self.__save_checkpoint__({
						'epoch': epoch + 1,
						'state_dict': self.model.state_dict(),
						'loss': loss,
						'optimizer' : self.optimizer.state_dict(),
						}, 'best')
					best_loss = loss
					logging.info('    Saved Best model with  \n{} \n'.format(loss))

				self.__save_checkpoint__({
						'epoch': epoch + 1,
						'state_dict': self.model.state_dict(),
						'loss': best_loss,
						'optimizer' : self.optimizer.state_dict(),
						}, '')
						
				scheduler.step()
				
				t.set_postfix(gpu = torch.cuda.max_memory_allocated() / 1024**3, loss = self.args.loss, best_loss = best_loss)
				t.update()
			

		gc.collect()
		logging.info('Training finalized with best average loss {}\n'.format(best_loss))
		return best_loss
		
	def __save_checkpoint__(self, state, checkpoint_type):
		checkpointpath, checkpointfile = get_checkpointname(	self.args, 
									checkpoint_type, 
									self.CViter)
		if not os.path.isdir(checkpointpath):
			os.mkdir(checkpointpath)
			
		torch.save(state, checkpointfile)


	def __resume_checkpoint__(self, checkpoint_type):
		_, checkpointfile = get_checkpointname(self.args, checkpoint_type, self.CViter)
		
		if not os.path.isfile(checkpointfile):
			return 0, inf
		else:
			logging.info("Loading checkpoint {}".format(checkpointfile))
			checkpoint = torch.load(checkpointfile, weights_only = False)
			start_epoch = checkpoint['epoch']
			loss = checkpoint['loss']
			self.model.load_state_dict(checkpoint['state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer'])
				
			return start_epoch, loss
			
	def __learning_rate_decay__(self, optimizer, decay_rate):
		if decay_rate < 1:
			for param_group in optimizer.param_groups:
				param_group['lr'] = param_group['lr'] * decay_rate
