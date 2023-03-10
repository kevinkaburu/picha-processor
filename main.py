import sys
import os
import boto3
import urllib.request as ur
import numpy as np
import requests
import cv2
import io
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pathlib import Path
import shutil
import mysql.connector
import time



from dotenv import load_dotenv


def imageProcessor(uploadID,DBConnection):
    # do something with uploadID
    s3 = boto3.resource('s3',  
    endpoint_url = os.getenv("endpoint_url"),
    aws_access_key_id = os.getenv("aws_access_key_id"),
    aws_secret_access_key = os.getenv("aws_secret_access_key"),
    region_name = os.getenv("region_name")
    )

    Userdir ="{}/raw".format(uploadID)
    bucket_name = os.getenv("bucket_name")
    bucket = s3.Bucket(bucket_name)
    #create directory
    dirpath = Path('processed/') / '{}'.format(uploadID)
    dirpath.mkdir(parents=True, exist_ok=True)
    PublicUrls = []


    for item in bucket.objects.filter(Prefix=Userdir):
        url = create_presigned_url(bucket_name, item.key)
        #uploadName = item.key get strign after /
        uploadName = item.key.split("/", 1)[1]
        print("Processing UploadID: {} | Image: {}".format(uploadID,uploadName))
        processedUrl = processImage(url,uploadID,uploadName, DBConnection,bucket_name,s3)
        if processedUrl:
            PublicUrls.append(processedUrl)

    updateUploadDB(uploadID,DBConnection)
    return PublicUrls


    

def create_presigned_url(bucket_name, object_name, expiration=3600):
    """Generate a presigned URL to share an S3 object

    :param bucket_name: string
    :param object_name: string
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Presigned URL as string. If error, returns None.
    """

    # Generate a presigned URL for the S3 object
    s3_client = boto3.client('s3',  
    endpoint_url = os.getenv("endpoint_url"),
    aws_access_key_id = os.getenv("aws_access_key_id"),
    aws_secret_access_key = os.getenv("aws_secret_access_key"),
    region_name = os.getenv("region_name")
    )

    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        print(e)
        return None

    # The response contains the presigned URL
    #string replace to get the url
    
    return response


def processImage(url, uploadID, uploadName, DBConnection, bucket_name, s3):
    # Load the image
    req = ur.urlopen(url)
    f = io.BytesIO(req.read())
    register_heif_opener()
    pilimage = Image.open(f)
    # take care of rotation
    pilimage = ImageOps.exif_transpose(pilimage)
    #get image name
    uploadName = uploadName.split(".", 1)[0]
    uploadName = uploadName.split("/", 1)[1]
    print("UploadID: {} | imageID: {}  Mode: {}".format(uploadID, uploadName, pilimage.mode))
    if pilimage.mode != "RGB":
        print("UploadID: {} | imageID: {}  Converting to RGB".format(uploadID, uploadName))
        pilimage = pilimage.convert("RGB")
    #resize image
    print("UploadID: {} | imageID: {}  Resizing image from {}x{}".format(uploadID, uploadName, pilimage.size[0], pilimage.size[1]))
    #if image is larger than 1024x1024 resize it
    if pilimage.size[0] > 1024 or pilimage.size[1] > 1024:
        pilimage.thumbnail((1024, 1024), Image.LANCZOS)
    cv2_img = np.array(pilimage)
    image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

    # Use a deep learning model to detect objects within the image
    model = cv2.dnn.readNetFromCaffe(os.getenv("deploy_prototype"), os.getenv("caffe_model"))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300,300), (103.93, 116.77, 123.68)) 

    model.setInput(blob)
    detections = model.forward()
    faces2 = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces2.append([startX, startY, endX - startX, endY - startY])

    print("UploadID: {} | imageID: {} Found {} faces!".format(uploadID, uploadName, len(faces2)))
    if len(faces2) > 0:
        largest_face = max(faces2, key=lambda x: x[2] * x[3])
        #get coordinates of largest face
        x1, y1, w, h = largest_face
        center_x = x1 + w // 2
        center_y = y1 + h // 2
        #set the dimensions of the crop box
        crop_size = 512
        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)
        right = min(image.shape[1], center_x + crop_size // 2)
        bottom = min(image.shape[0], center_y + crop_size // 2)
        # Crop the image
        cropped_image = image[top:bottom, left:right]
        print("UploadID: {} | imageID: {} Cropped image to {}x{} FROM: {}x{}".format(uploadID, uploadName, cropped_image.shape[1], cropped_image.shape[0],image.shape[1], image.shape[0]))

    else:
        return None

    h, w = cropped_image.shape[:2]
    min_size = np.amin([h,w])

    # # Centralize and crop
    # crop_img = cropped_image[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized_img = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)

    # Save the resized image as PNG
    #upload name remove extension
    newPng ="processed/{}/{}.png".format(uploadID,uploadName)
    cv2.imwrite(newPng, resized_img)
    #upload to s3
    s3.meta.client.upload_file(newPng, bucket_name, '{}/processed/{}.png'.format(uploadID,uploadName))
    #update DB
    publicUrl='{}{}/processed/{}.png'.format(os.getenv('bucket_public_url'),uploadID,uploadName)
    updateUploadImgDB(uploadName, publicUrl,DBConnection)
    return publicUrl

def updateUploadDB(uploadID,DBConnection):   
    #update database
    mycursor = DBConnection.cursor()
    sql = "UPDATE upload SET status=5 WHERE upload_id = {}".format(uploadID)
    mycursor.execute(sql)
    DBConnection.commit()

#update database set table upload_image bucket_url 
def updateUploadImgDB(uploadID, bucket_url,DBConnection):   
    #update database
    mycursor = DBConnection.cursor()
    sql = "UPDATE upload_image SET url = %s, status=5 WHERE upload_image_id = %s"
    mycursor.execute(sql, (bucket_url, uploadID))
    DBConnection.commit()
    print("ImageID:{} update record affected: {} url: {}".format(uploadID,mycursor.rowcount,bucket_url))

#init model training
def initModelTraining(transactionID,uploadID,images,DBConnection):
    #get train_type from database
    selectTrainType = "SELECT tt.name FROM upload u inner join train_type tt using(train_type_id) WHERE u.upload_id = {}".format(uploadID)
    mycursor = DBConnection.cursor()
    mycursor.execute(selectTrainType)
    myresult = mycursor.fetchall()
    train_type = myresult[0][0]
    #select train_model_id from database
    selectTrainModel = "SELECT train_model_id FROM transaction WHERE transaction_id= {}".format(transactionID)
    mycursor.execute(selectTrainModel)
    myresult = mycursor.fetchall()
    train_model_id = myresult[0][0]
    mycursor.close()
    classType = "man"
    training_type ="men"
    if train_type == "Female":
         training_type ="female"
         classType = "woman"


    #prepare json to send to model training
    payload ={
    "key": "{}".format(os.getenv('model_training_key')),
    "instance_prompt": "sks",
    "class_prompt" : "a photo of a {}".format(classType),
    "base_model_id" : "dream-shaper-8797",
    "images": images,
    "seed": "0",
    "training_type": "{}".format(training_type),
    "max_train_steps": "2000",
    "webhook": "{}?transaction_id={}".format(os.getenv('model_training_webhook'),transactionID)
    }
    #send request to model training
    r = requests.post(os.getenv('stablediffusionapi_training_url'), json=payload)
    print("UploadID: {} | Model training response: {}".format(uploadID,r.json()))
    response = r.json()
    #update database
    mycursor = DBConnection.cursor()
    sqlUpdateTransaction = "UPDATE transaction SET status=2 WHERE transaction_id = {}".format(transactionID)
    mycursor.execute(sqlUpdateTransaction)
    #update database
    myother = DBConnection.cursor()
    sqlUpdateUpload = "UPDATE train_model SET status='{}',external_model_id='{}' WHERE train_model_id = {}".format(response.get('messege'),response.get('training_id'),train_model_id)
    myother.execute(sqlUpdateUpload)
    DBConnection.commit()
    mycursor.close()
    myother.close()

if __name__ == "__main__":
    uploadID = sys.argv[1]
    transactionID = sys.argv[2]
    start = time.time()
    print("-----\nStarting uploadID: {} transactionID: {}".format(uploadID,transactionID))
    load_dotenv()
    mydb = mysql.connector.connect(
    host=os.getenv("host"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    database=os.getenv("database")
    )
    #get upload images
    fileNames = imageProcessor(uploadID,mydb)
     #delete directory
    dirpath = Path('processed/') / '{}'.format(uploadID)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    #start model training
    print("uploadID: {} start model training....\n".format(uploadID))
    initModelTraining(transactionID,uploadID,fileNames,mydb)

    mydb.close()
    end = time.time()
    #print time taken by uploadID
    print("-----\nEnd Upload ID: {} took: {} mins".format(uploadID, (end - start)/60))
    
