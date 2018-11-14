# Can't be run unless your AWS ID has permission to use the Rekognition API.
# Well it'll run, just not very successfully!
import boto3
from os import listdir
from os.path import isfile, join, basename
from io import BytesIO
from PIL import Image
import json

def download_s3_resource(s3, bucket_name, key):
    file_contents = s3.Object(bucket_name, key).get()["Body"]

    return (file_contents)


def download_image(s3, bucket_name, key):
    fake_file = BytesIO(download_s3_resource(s3, bucket_name, key).read())

    return (Image.open(fake_file))


client = boto3.client('rekognition', region_name="eu-west-1")


# TODO command line args!


# List images w/ faces
bucket_name = "smile-detection-test"
face_path = "faces/"

s3 = boto3.resource('s3', region_name='eu-west-1')
bucket = s3.Bucket(bucket_name)

face_files = []
for obj in bucket.objects.filter(Prefix=face_path):
    if !(obj.key.endswith(".png") or obj.key.endswith(".jpg")):
        continue

    test_file_key = obj.key
    if type(test_file_key) == str:
        face_files.append(test_file_key)


# We need to record "smile probability" for each face
image_to_smile = dict()

images_len = len(image_ref.keys())
cnt = 0
for photo_key in face_files:
    cnt++
    image_ref = dict()
    image_ref["S3Object"] = {"Bucket": bucket_name, "Name": photo_key}

    if cnt % 100 == 0:
        print("Processed " + str(cnt) + " images out of" + str(images_len))

    #    print("Detecting faces for key [" + photo_key + "]")

    # Detect faces
    detection_result = client.detect_faces(Image = image_ref)
    if 'FaceDetails' in detection_result:
        if len(detection_result['FaceDetails']) > 0:
            # TODO Smiling?  Put image page + smile probability in image_to_smile json
            image_to_smile[photo_key] = detection_result

# Here we save out what to what degree each face is smiling
with open("image_to_smile.json", "w") as fh:
    json.dump(image_to_smile, fh)

