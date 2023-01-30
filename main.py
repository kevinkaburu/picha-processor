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
    zipObj = ZipFile('processed/{}/zip/{}.zip'.format(uploadID,uploadID), 'w')

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
    s3.meta.client.upload_file('processed/{}/zip/{}.zip'.format(uploadID,uploadID), bucket_name, 'zip/{}.zip'.format(uploadID))
    dirpath = Path('processed/') / '{}'.format(uploadID)
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)


    

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

    # Resize the image to 512x512 pixels
    resized_img = cv2.resize(image, (512, 512))

    # Save the resized image as PNG
    #upload name remove extension
    uploadName = uploadName.split(".", 1)[0]
    newPng ="processed/{}/{}.png".format(uploadID,uploadName)
    cv2.imwrite(newPng, resized_img)
    zipObj.write(newPng)


if __name__ == "__main__":
    load_dotenv()
    #main.py 16
    imageProcessor(sys.argv[1])