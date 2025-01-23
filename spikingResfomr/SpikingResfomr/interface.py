import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
import models.spikingresformer
from timm.models import create_model


warnings.filterwarnings('ignore')

def main(args):

   

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device='cuda')
    face_detector.eval()

    video_list,target_list=init_ff()

    
    print('video_list:' + str(len(video_list)))
    output_list=[]
    for filename in tqdm(video_list):
        
        try:

            face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)
            model=create_model(
            'spikingresformer_l',
            checkpoint_path='logs/temp/checkpoint_max_acc1.pth',

            num_classes=2

            ).cuda()
            model.eval()
            
            with torch.no_grad():

                img=torch.tensor(face_list).to(device).float()/255
                if(len(face_list)!= 32):
                    predictionList=[]
                    try:
                        for im in img:
                            im = im.unsqueeze(0)
                            pred=model(im)
                            final_pred = pred[-1]
  
                            pred=final_pred.softmax(1)[:,1]
                            predictionList.append(pred.item())
                        pred=torch.tensor(predictionList).to(device)

                        
                    except Exception as e:
                        print(e)
                        exit()
                else:
                    pred=model(img)
                    final_pred = pred[-1]  
                    pred=final_pred.softmax(1)[:,1]

                """for i in range(len(idx_list)):
                    print(i)
                    print(img[i])
                    print(len(img[i]))
                    print(img.shape)
                    print(img[i].shape)
                    print("perform predic")
                    im=img[i].unsqueeze(0)
                    print(im.shape)

                    pred=model(im).sigmoid()
                    final_pred = pred[-1]
                    print(pred)
                    print(final_pred)
                    exit()
"""
            

                

            pred_list=[]
            idx_img=-1
            
            for i in range(len(pred)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res=np.zeros(len(pred_list))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
            pred=pred_res.mean()
        except Exception as e:
            print(e)
            pred=0.5
        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')
    

if __name__=='__main__':

    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',default='FF',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    args=parser.parse_args()

    main(args)
