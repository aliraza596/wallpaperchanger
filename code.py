try:
	import unzip_requirements
except ImportError:
	pass

import json
import base64
import torch
import torchvision
import torchvision.transforms
import numpy as np
import cv2
import boto3
import csv
import PIL.Image
from numpy import load
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import os
from os import path
import uuid
import requests
mybucket = 'modelfiless'
inputbucket = 'inputimagess'
wallpaperbucket = 'wallpaperssss'
outputbucket = 'processed.imagess'
name = str(uuid.uuid4()) + ".jpg"

def write_to_file(save_path, data):
  with open(save_path, "wb") as f:
    f.write(base64.b64decode(data))
    
def get_model_encoder():
	if path.exists("/tmp/encoder_epoch_20.pth"):
		file_path1 = os.path.join('/tmp/', 'encoder_epoch_20.pth')
	else:
		strKey = 'encoder_epoch_20.pth'
		strFile = '/tmp/encoder_epoch_20.pth'
		downloadFromS3(mybucket, strKey, strFile)
		file_path1 = os.path.join('/tmp/', 'encoder_epoch_20.pth')
	return file_path1

def get_model_decoder():
	if path.exists("/tmp/decoder_epoch_20.pth"):
		file_path2 = os.path.join('/tmp/', 'decoder_epoch_20.pth')
	else:
		strKey = 'decoder_epoch_20.pth'
		strFile = '/tmp/decoder_epoch_20.pth'
		downloadFromS3(mybucket, strKey, strFile)
		file_path2 = os.path.join('/tmp/', 'decoder_epoch_20.pth')
	return file_path2
	
def s3_upload_image(filename):
	client = boto3.client('s3')
	client.upload_file(filename, outputbucket, name)
	
def downloadFromS3(strBucket, strKey,strFile):
	s3_client = boto3.client('s3')
	s3_client.download_file(strBucket, strKey, strFile)	

def get_input():
	if path.exists("/tmp/room.jpeg"):
		file_path3 = os.path.join('/tmp/', 'room.jpeg')
	else:
		strKey1 = 'room.jpeg'
		strFile1 = '/tmp/room.jpeg'
		downloadFromS3(inputbucket, strKey1, strFile1)
		file_path3 = os.path.join('/tmp/', 'room.jpeg')
	return file_path3
	
def get_wallpaper():
	if path.exists("/tmp/wallpaper.jpg"):
		file_path4 = os.path.join('/tmp/', 'wallpaper.jpg')
	else:
		strKey2 = 'wallpaper.jpg'
		strFile2 = '/tmp/wallpaper.jpg'
		downloadFromS3(wallpaperbucket, strKey2, strFile2)
		file_path4 = os.path.join('/tmp/', 'wallpaper.jpg')
	return file_path4
	
# load array
colors = load('colors.npy')
names = {}
with open('object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]
        
def network(encoder, decoder):
	# Network Builders
	net_encoder = ModelBuilder.build_encoder(
		arch='resnet50dilated',
		fc_dim=2048,
		weights=encoder)
	net_decoder = ModelBuilder.build_decoder(
		arch='ppm_deepsup',
		fc_dim=2048,
		num_class=150,
		weights=decoder,
		use_softmax=True)
	crit = torch.nn.NLLLoss(ignore_index=-1)
	segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
	segmentation_module.eval()
	segmentation_module
	return segmentation_module
    
def normalize(image_path):
	# Load and normalize one image as a singleton tensor batch
	pil_to_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	pil_image = PIL.Image.open(image_path).convert('RGB')
	img_original = np.array(pil_image)
	img_data = pil_to_tensor(pil_image)
	singleton_batch = {'img_data': img_data[None]}
	output_size = img_data.shape[1:]
	return singleton_batch, output_size, img_original
    
def predict(segmentation_module, singleton_batch, output_size):
	# Run the segmentation at the highest resolution.
	with torch.no_grad():
		scores = segmentation_module(singleton_batch, segSize=output_size)
	_, pred = torch.max(scores, dim=1)
	pred = pred.cpu()[0].numpy()
	return pred
    
class Montage(object):
    def __init__(self,initial_image):
        self.montage = initial_image
        self.x,self.y = self.montage.shape[:2]

    def append(self,image):
        image = image[:,:,:3]
        x,y = image.shape[0:2]
        new_image = cv2.resize(image,(int(y*float(self.x)/x),self.x))
        self.montage1 = np.hstack((self.montage,new_image))
        self.montage2 = np.vstack((self.montage1,self.montage1))
        self.montage3 = np.hstack((self.montage2,self.montage2))
        self.montage4 = np.vstack((self.montage3,self.montage3))
        self.montage5 = np.hstack((self.montage4,self.montage4))
        self.montage6 = np.vstack((self.montage5,self.montage5))

        return self.montage6
        
def visualize(img, pred, f, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        #print(pred)
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)
    #print(pred_color.shape)
    gray_img = cv2.cvtColor(pred_color, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    mask2 = cv2.bitwise_not(thresh)
    gcMask = mask2.copy()
    gcMask[gcMask > 0] = cv2.GC_PR_FGD
    gcMask[gcMask == 0] = cv2.GC_BGD
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    #if output is bad, then try changing the iterCount from 0 to 10 and check where the result comes best. I set default as 5.
    (gcMask, bgModel, fgModel) = cv2.grabCut(img, gcMask,
                                             None, bgModel, fgModel, iterCount=5,
                                             mode=cv2.GC_INIT_WITH_MASK)
    outputMask = np.where(
        	(gcMask == cv2.GC_BGD) | (gcMask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")
    dilation = cv2.dilate(outputMask,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    median = cv2.medianBlur(closing,5)
    # apply a bitwise AND to the image using our mask generated
	# by GrabCut to generate our final output image
    output = cv2.bitwise_and(img, img, mask=median)

    mask_inv = cv2.bitwise_not(median)
    mo = cv2.resize(f, (img.shape[1],img.shape[0]))
    #print(mo.shape)
    #print(img.shape)
    img1_bg = cv2.bitwise_and(img,img,mask = mask_inv)
    img2_fg = cv2.bitwise_and(mo,mo,mask = median)
    dst = cv2.add(img1_bg,img2_fg)
    result = PIL.Image.fromarray(dst)
    return result
    
def lambda_handler(event, context):
	modelencoder_load = get_model_encoder()
	modeldecoder_load = get_model_decoder()
	write_to_file("/tmp/photo.jpg", event["body"])
	wallpaper = get_wallpaper()
	
	segmentation_module = network(modelencoder_load,modeldecoder_load)
	singleton_batch, output_size, img_original = normalize("/tmp/photo.jpg")
	pred = predict(segmentation_module, singleton_batch, output_size)
	
	w = cv2.imread(wallpaper)
	m = Montage(w)
	f = m.append(w)
	
	predicted_classes = np.bincount(pred.flatten()).argsort()[::-1]
	for c in predicted_classes[:1]:
		result = visualize(img_original, pred, f, c)
	
	filename = '/tmp/output.jpg'
	result.save(filename)
	s3_upload_image(filename)
	show = cv2.imread(filename)
	s3_client = boto3.client('s3')
	url = s3_client.generate_presigned_url('get_object', Params= {'Bucket': "processed.imagess","Key": name})

	return {
		'statusCode': 200,
		'body': json.dumps(url)
	}
		
	

	