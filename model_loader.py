import os
import sys
import torch # pyright: ignore[reportMissingImports]
import logging
import torch.nn as nn # pyright: ignore[reportMissingImports]

in_channels = 1

def loadModel(netname, channels):
	Netpath = 'Model'
	Netfile = os.path.join(Netpath, netname)
	Netfile = os.path.join(Netfile, netname + '.py')
	assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
	netname = netname.lower()
	if netname == 'jin': 
		return loadjin(channels)
	elif netname == 'jinplus': 
		return loadjinplus(channels)
	elif netname == 'jinpp': 
		return loadjinpp(channels)
	elif netname == 'ujin': 
		return loadunjin(channels)
	elif netname == 'jinppvit': 
		return loadjinppvit(channels)
	elif netname == 'unet': 
		return loadunet(channels)
	elif netname == 'nnunet': 
		return loadnnunet(channels)
	elif netname == 'unett': 
		return loadunett(channels)
	elif netname == 'msegnet': 
		return loadmsegnet(channels)
	elif netname == 'segformer':
		return loadsegformer(channels)
	elif netname == 'medt':
		return loadmedt(channels)
	elif netname == 'transunet':
		return loadtransunet(channels)
	elif netname == 'deeplabv3':
		return loaddeeplabv3(channels)
	elif netname == 'lightawnet':
		return loadlightawnet(channels)
	elif netname == 'medformer':
		return loadmedformer(channels)
	elif netname == 'missformer':
		return loadmissformer(channels)
	elif netname == 'mobilenetv2_unet':
		return loadmobilenetv2_unet(channels)
	elif netname == 'mobilenetv3_unet':
		return loadmobilenetv3_unet(channels)
	elif netname == 'swinunet':
		return loadswinunet(channels)
	else:
		logging.warning("No model with the name {} found, please check your spelling.".format(netname))
		sys.exit()


def get_model_list(netname = ''):
	netname = netname.lower()
	net_list = [
		'Segformer', 'Unet', 'nnUnet', 'UnetT', 'MedT', 'MSegnet', 'Jin', 'JinPlus', 'JinPP', 'UJin', 'JinPPViT', 'Transunet',
		'Deeplabv3', 'Lightawnet', 'Medformer', 'Missformer', 'Mobilenetv2_unet', 'Mobilenetv3_unet', 'Swinunet'
	]
	

	if netname == '':
		return [item.lower() for item in net_list]
	if netname in [item.lower() for item in net_list]:
		return [netname]

	logging.warning("No model with the name {} found, please check your spelling.".format(netname))
	logging.warning("Net List:")
	for net in net_list:
		logging.warning("	{}}".format(net))
	sys.exit()

def loaddeeplabv3(channels):
	from Model.deeplabv3.deeplabv3 import DeepLabV3
	logging.info("Loading DeepLabV3 Model")
	return DeepLabV3(in_channels = in_channels, num_classes = channels)
	
def loadlightawnet(channels):
	from Model.lightawnet.lightawnet import LightAWNet
	logging.info("Loading LightAwnet Model")
	return LightAWNet(in_channels = in_channels, num_classes = channels)

def loadmedformer(channels):
	from Model.medformer.medformer import MedFormer
	logging.info("Loading Medformer Model")
	return MedFormer(in_channels = in_channels, num_classes = channels)

def loadmissformer(channels):
	from Model.missformer.missformer import MISSFormer
	logging.info("Loading Missformer Model")
	return MISSFormer(in_channels = in_channels, num_classes = channels)

def loadmobilenetv2_unet(channels):
	from Model.mobilenetv2_unet.mobilenetv2_unet import MobileNetV2_UNet
	logging.info("Loading MobileNetV2_UNet Model")
	return MobileNetV2_UNet(in_channels = in_channels, num_classes = channels)

def loadmobilenetv3_unet(channels):
	from Model.mobilenetv3_unet.mobilenetv3_unet import MobileNetV3_UNet
	logging.info("Loading MobileNetV3_UNet Model")
	return MobileNetV3_UNet(in_channels = in_channels, num_classes = channels)

def loadswinunet(channels):
	from Model.swinunet.swinunet import SwinUNet
	logging.info("Loading SwinUNet Model")
	return SwinUNet(in_channels = in_channels, num_classes = channels)
	
def loadtransunet(channels):
	from Model.transunet.transunet import TransUNet
	logging.info("Loading TransUNet Model")
	return TransUNet(in_channels = in_channels, num_classes = channels)
	
def loadjin(channels):
	from Model.jin.jin import JinNet
	logging.info("Loading JinNet Model")
	return JinNet(in_channels = in_channels, num_classes = channels)
	
def loadunjin(channels):
	from Model.ujin.ujin import UJintransformer
	logging.info("Loading UJintransformer Model")
	return UJintransformer(in_channels = in_channels, num_classes = channels)

def loadjinppvit(channels):
	from Model.jinppvit.jinppvit import JinPPViT
	logging.info("Loading JinPPViT Model")
	return JinPPViT(in_channels = in_channels, num_classes = channels)

def loadjinplus(channels):
	from Model.jinplus.jinplus import JinPlus
	logging.info("Loading JinPlus Model")
	return JinPlus(in_channels = in_channels, num_classes = channels)
	
def loadjinpp(channels):
	from Model.jinpp.jinpp import JinPP
	logging.info("Loading JinPP Model")
	return JinPP(in_channels = in_channels, num_classes = channels)

def loadsegformer(channels):
	from Model.segformer.segformer import Segformer
	logging.info("Loading Segformer Model")
	return Segformer(in_channels = in_channels, num_classes = channels)

def loadmedt(channels):
	from Model.medt.medt import MedicalTransformer
	logging.info("Loading MedicalTransformer Model")
	return MedicalTransformer(in_channels = in_channels, num_classes = channels)

def loadmsegnet(channels):
	from Model.msegnet.msegnet import MSegNet
	logging.info("Loading MSegnet Model")
	return MSegNet(in_channels = in_channels, num_classes = channels)
	
def loadunet(channels):
	from Model.unet.unet import Unet
	logging.info("Loading Unet Model")
	return Unet(in_channels = in_channels, num_classes = channels)

def loadnnunet(channels):
	from Model.nnunet.nnunet import nnUNet
	logging.info("Loading nnUNet Model")
	return nnUNet(in_channels = in_channels, num_classes = channels)

def loadunett(channels):
	from Model.unett.unett import UNetTransformer
	logging.info("Loading UnetTransformer Model")
	return UNetTransformer(in_channels = in_channels, num_classes = channels)


def weight_ini(m):
	torch.manual_seed(230)
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
		m.reset_parameters()
	


