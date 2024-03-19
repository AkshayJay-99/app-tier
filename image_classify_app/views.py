from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from asgiref.sync import async_to_sync
import os
import csv
import sys
import torch
from PIL import Image
from .facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
import boto3

local_folder = 'image_classify_app/static/images'
data_path = 'image_classify_app/static/images/data/data.pt'
# Create your views here.

s3 = boto3.client('s3')
sqs = boto3.client('sqs', region_name='us-east-1')
req_queue_name = '1229059769-req-queue'
req_queue_url = 'https://sqs.us-east-1.amazonaws.com/533267431319/1229059769-req-queue'
out_bucket_name = '1229059769-out-bucket'

ec2 = boto3.client('ec2')
ec2_instance_ids = ['i-076ff9cd3fa9d5c78', 'i-072e2192ebef2c6c4', 'i-09c3e439d0091156a', 'i-0dbb687d7ddabcbd3', 'i-0a6d49fbd52d8aefa', 'i-0b313197f356d0f2a', 'i-07975344fb98c50f2', 'i-027b01d4637907976', 'i-0669ce2be7099542f', 'i-041898d207a9b6a7e', 'i-001d1a0fbf0db9611', 'i-01b28d1e1b5fc249d', 'i-01f01b6941f8ce643', 'i-0114d1c66d351d304', 'i-02493c472fd4c2b95', 'i-0e89e10f9d19d2562', 'i-0daf89b037b464d8f', 'i-0ba531c5cd85cec75']


mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

@csrf_exempt
def img_classify(request):
    if request.method == 'POST' and 'imageFile' in request.FILES:
        input_file = request.FILES['imageFile']
        input_file_name = input_file.name.split('.')[0]
        file_name = input_file.name
        local_folder_path = local_folder
        local_file_path = local_folder_path+'/'+file_name
        # Save the file to the local folder
        with open(local_file_path, 'wb') as local_file:
            local_file.write(input_file.read())
        result = img_recog(local_file_path,data_path)
        upload_to_s3_outbucket(result[0], out_bucket_name, input_file_name)
       
        return HttpResponse(f"{input_file_name}:{result[0]}")
    else:
        return HttpResponse("Invalid request")
    
def img_recog(file_path,data_path):
     # getting embedding matrix of the given img
    img = Image.open(file_path)
    face, prob = mtcnn(img, return_prob=True) # returns cropped face and probability
    emb = resnet(face.unsqueeze(0)).detach() # detech is to make required gradient false

    saved_data = torch.load(data_path) # loading data.pt file
    embedding_list = saved_data[0] # getting embedding data
    name_list = saved_data[1] # getting list of names
    dist_list = [] # list of matched distances, minimum distance is used to identify the person

    for idx, emb_db in enumerate(embedding_list):
        dist = torch.dist(emb, emb_db).item()
        dist_list.append(dist)

    idx_min = dist_list.index(min(dist_list))
    return (name_list[idx_min], min(dist_list))

def upload_to_s3_outbucket(value, s3_bucket, s3_key):
    #s3.upload_file(local_path, s3_bucket, s3_key)
    s3.put_object(Body=value, Bucket=s3_bucket, Key=s3_key)
    
def del_messages_from_sqs():
    response = sqs.receive_message(QueueUrl=req_queue_url, MaxNumberOfMessages=1)
    messages = response.get('Messages', [])

    if messages:
        msg = messages[0]
        receipt_handle = msg['ReceiptHandle']
        sqs.delete_message(QueueUrl=req_queue_url, ReceiptHandle=receipt_handle)


