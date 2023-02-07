import sys
import os
import boto3
import urllib.request as ur
import numpy as np
import cv2
import io
from PIL import Image
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


    for item in bucket.objects.filter(Prefix=Userdir):
        url = create_presigned_url(bucket_name, item.key)
        #uploadName = item.key get strign after /
        uploadName = item.key.split("/", 1)[1]
        print("Processing UploadID: {} | Image: {}".format(uploadID,uploadName))
        processImage(url,uploadID,uploadName, DBConnection,bucket_name,s3)

    updateUploadDB(uploadID,DBConnection)


    

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


def processImage(url,uploadID,uploadName,DBConnection,bucket_name,s3):
    # Load the image
    req = ur.urlopen(url)
    f = io.BytesIO(req.read())
    register_heif_opener()
    pilimage = Image.open(f)
    uploadName = uploadName.split(".", 1)[0]
    uploadName = uploadName.split("/", 1)[1]
    print("UploadID: {} | imageID: {}  Mode:{} ".format(uploadID,uploadName,pilimage.mode))
    if pilimage.mode != "RGB":
        pilimage = pilimage.convert("RGB")
    
    
    cv2_img = np.array(pilimage)
    image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a cascading classifier to detect objects within the image
    face_cascade = cv2.CascadeClassifier(os.getenv("face_cascade"))

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=8, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    #load model
    model = cv2.dnn.readNetFromCaffe(os.getenv("deploy_prototype"), os.getenv("caffe_model"))
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    model.setInput(blob)
    detections = model.forward()
    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.8:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append([startX, startY, endX - startX, endY - startY])

    print("UploadID: {} | imageID: {} Found {} faces!".format(uploadID,uploadName,len(faces)))
    if len(faces) > 0:
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        #get coordinates of largest face
        x1, y1, w, h = largest_face
        x2 = x1 + w
        y2 = y1 + h
        # Add more space to the top, bottom, left and right of the face
        space = 0.50
        x1 -= int(space * (x2 - x1))
        x2 += int(space * (x2 - x1))
        y1 -= int(space * (y2 - y1))
        y2 += int(space * (y2 - y1))
        # Ensure that the cropped area stays within the bounds of the image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        # Crop the image
        cropped_image = image[y1:y2, x1:x2]
    else:
        # If no faces are detected, crop the image to a square centered around the center of the image
        print("UploadID: {} | imageID: {} No faces detected".format(uploadID,uploadName))
        return

    resized_img = cv2.resize(cropped_image, (512, 512), interpolation=cv2.INTER_AREA)

    # Save the resized image as PNG
    #upload name remove extension
  
    newPng ="processed/{}/{}.png".format(uploadID,uploadName)
    cv2.imwrite(newPng, resized_img)
    #upload to s3
    s3.meta.client.upload_file(newPng, bucket_name, '{}/processed/{}.png'.format(uploadID,uploadName))
    #update DB
    updateUploadImgDB(uploadName, '{}{}/processed/{}.png'.format(os.getenv('bucket_public_url'),uploadID,uploadName),DBConnection)

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


if __name__ == "__main__":
    start = time.time()
    load_dotenv()
    mydb = mysql.connector.connect(
    host=os.getenv("host"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    database=os.getenv("database")
    )

    uploadID = sys.argv[1]
    imageProcessor(uploadID,mydb)
    mydb.close()
    #delete directory
    dirpath = Path('processed/') / '{}'.format(uploadID)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    end = time.time()
    #print time taken by uploadID
    print("-----\nUpload ID: {} took: {} mins".format(uploadID, (end - start)/60))
    
