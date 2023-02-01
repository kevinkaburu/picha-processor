import sys
import os
from zipfile import ZipFile
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



from dotenv import load_dotenv


def imageProcessor(uploadID):
    # do something with uploadID
    s3 = boto3.resource('s3',  
    endpoint_url = os.getenv("endpoint_url"),
    aws_access_key_id = os.getenv("aws_access_key_id"),
    aws_secret_access_key = os.getenv("aws_secret_access_key"),
    region_name = os.getenv("region_name")
    )

    Userdir ="{}".format(uploadID)
    bucket_name = os.getenv("bucket_name")
    bucket = s3.Bucket(bucket_name)
    #create directory
    dirpath = Path('processed/') / '{}'.format(uploadID)
    dirpath.mkdir(parents=True, exist_ok=True)
    dirpath = Path('processed/') / '{}/zip'.format(uploadID)
    dirpath.mkdir(parents=True, exist_ok=True)

    #create zip file
    zipObj = ZipFile('{}.zip'.format(uploadID), 'w')

    print('Objects:')
    for item in bucket.objects.filter(Prefix=Userdir):
        url = create_presigned_url(bucket_name, item.key)
        #uploadName = item.key get strign after /
        uploadName = item.key.split("/", 1)[1]
        print("bucket: {}| file: {}| Name: {}| url:{}".format(bucket_name,item.key,uploadName,url))
        processImage(url,uploadID,uploadName, zipObj)
        print('\n ', url)

    zipObj.close()
    #upload zip to s3
    s3.meta.client.upload_file('{}.zip'.format(uploadID), bucket_name, 'zip/{}.zip'.format(uploadID))
    #delete directory
    dirpath = Path('processed/') / '{}'.format(uploadID)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    #delete zip file
    os.remove('{}.zip'.format(uploadID))
    #update database
    bucket_url = create_presigned_url(bucket_name, 'zip/{}.zip'.format(uploadID),(604800-1))
    updateDatabase(uploadID, bucket_url)


    

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


def processImage(url,uploadID,uploadName,zipObj):
    # Load the image
    req = ur.urlopen(url)
    f = io.BytesIO(req.read())
    register_heif_opener()
    pilimage = Image.open(f)
    print("Mode: ", pilimage.mode)
    if pilimage.mode != "RGB":
        pilimage = pilimage.convert("RGB")
    
    
    cv2_img = np.array(pilimage)
    image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use a cascading classifier to detect objects within the image
    face_cascade = cv2.CascadeClassifier(os.getenv("face_cascade"))
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        #get coordinates of largest face
        x1, y1, w, h = largest_face
        x2 = x1 + w
        y2 = y1 + h
        # Add more space to the top, bottom, left and right of the face
        space1 = 0.35
        space2 = 0.45
        x1 -= int(space1 * (x2 - x1))
        x2 += int(space2 * (x2 - x1))
        y1 -= int(space1 * (y2 - y1))
        y2 += int(space2 * (y2 - y1))
        # Ensure that the cropped area stays within the bounds of the image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        # Crop the image
        cropped_image = image[y1:y2, x1:x2]
    else:
        # If no faces are detected, crop the image to a square centered around the center of the image
        cropped_image = image

    resized_img = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)

    # Save the resized image as PNG
    #upload name remove extension
    uploadName = uploadName.split(".", 1)[0]
    newPng ="processed/{}/{}.png".format(uploadID,uploadName)
    cv2.imwrite(newPng, resized_img)
    zipObj.write(newPng, arcname="{}.png".format(uploadName))

#update database set table upload bucket_url to zip file 
def updateDatabase(uploadID, bucket_url):   
    mydb = mysql.connector.connect(
    host=os.getenv("host"),
    user=os.getenv("user"),
    password=os.getenv("password"),
    database=os.getenv("database")
    )
    #update database
    mycursor = mydb.cursor()
    sql = "UPDATE upload SET bucket_url = %s, status=5 WHERE upload_id = %s"
    mycursor.execute(sql, (bucket_url, uploadID))
    mydb.commit()
    print(mycursor.rowcount, "record(s) affected")


if __name__ == "__main__":
    load_dotenv()
    #main.py 16
    imageProcessor(sys.argv[1])
