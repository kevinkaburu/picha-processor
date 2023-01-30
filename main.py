import sys
import os
import boto3
import urllib.request as ur
import numpy as np
import cv2
import io
from PIL import Image

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

    print('Objects:')
    for item in bucket.objects.filter(Prefix=Userdir):
        url = create_presigned_url(bucket_name, item.key)
        #uploadName = item.key get strign after /
        uploadName = item.key.split("/", 1)[1]
        print("bucket: {}| file: {}| Name: {}| url:{}".format(bucket_name,item.key,uploadName,url))
        processImage(url,uploadID,uploadName)
        print('\n ', url)

    

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


def processImage(url,uploadID,uploadName):
    # Load the image
    req = ur.urlopen(url)
    f = io.BytesIO(req.read())
    pilimage = Image.open(f)
    print("Mode: ", pilimage.mode)
    if pilimage.mode != "RGB":
        pilimage = pilimage.convert("RGB")

    
    cv2_img = np.array(pilimage)
    image = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the face detection model
    face_cascade = cv2.CascadeClassifier(os.getenv("face_cascade"))

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # # Check if only one face is present in the image
    # print("Number of faces detected: ", len(faces))
    # if len(faces) < 1:
    #     print("Error: Atlease one face must be present in the image")
    #     exit()
    # elif len(faces) > 2:
    #     print("Error: Only one face must be present in the image")
    #     exit()


    # Crop the image around the face
    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w]
        face_center_x, face_center_y = (x + w/2, y + h/2)
        x_offset, y_offset = (256 - face_center_x, 256 - face_center_y)
        y_coord = max(y_offset, 0)
        x_coord = max(x_offset, 0)
        face_img = image[y_coord:y_coord+512, x_coord:x_coord+512]
        face_area = w*h
        total_area = 512*512
        if (face_area/total_area) < 0.35:
           face_img = cv2.resize(face_img, (int(w*1.2), int(h*1.2)))
           face_img = face_img[int(h*0.1):int(h*1.1), int(w*0.1):int(w*1.1)]

    # Resize the image to 512x512 pixels
    resized_img = cv2.resize(face_img, (512, 512))

    # Save the resized image as PNG
    cv2.imwrite("processed/{}/{}.png".format(uploadID,uploadName), resized_img)


if __name__ == "__main__":
    load_dotenv()
    #main.py 16
    imageProcessor(sys.argv[1])