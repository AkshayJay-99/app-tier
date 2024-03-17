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
data_path = 'image_classify_app\static\images\data\data.pt'
# Create your views here.

s3 = boto3.client('s3')
sqs = boto3.client('sqs', region_name='us-east-1')
req_queue_name = '1229059769-req-queue'
req_queue_url = 'https://sqs.us-east-1.amazonaws.com/533267431319/1229059769-req-queue'
out_bucket_name = '1229059769-out-bucket'

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
        response = sqs.get_queue_attributes(
            QueueUrl=req_queue_url,
            AttributeNames=['ApproximateNumberOfMessages']
        )

        approximate_message_count = int(response['Attributes']['ApproximateNumberOfMessages'])
        if approximate_message_count>0:
            del_messages_from_sqs()
            
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


