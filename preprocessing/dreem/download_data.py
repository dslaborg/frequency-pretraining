"""
Helper script to download the DOD-O and DOD-H datasets from the S3 bucket. Currently, the script downloads the data
to the `~/data/dreem` directory.
Mostly copied from https://github.com/Dreem-Organization/dreem-learning-open/blob/master/download_data.py with some
modifications.
"""
import os
from os.path import expanduser

import boto3
import tqdm
from botocore import UNSIGNED
from botocore.client import Config

path = expanduser("~/data/dreem")

# make sure the directories exist
os.makedirs(path + "/dodo", exist_ok=True)
os.makedirs(path + "/dodh", exist_ok=True)

client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

bucket_objects = client.list_objects(Bucket="dreem-dod-o")["Contents"]
print("\n Downloading H5 files and annotations from S3 for DOD-O")
for bucket_object in tqdm.tqdm(bucket_objects):
    filename = bucket_object["Key"]
    client.download_file(
        Bucket="dreem-dod-o", Key=filename, Filename=path + "/dodo/{}".format(filename)
    )

bucket_objects = client.list_objects(Bucket="dreem-dod-h")["Contents"]
print("\n Downloading H5 files and annotations from S3 for DOD-H")
for bucket_object in tqdm.tqdm(bucket_objects):
    filename = bucket_object["Key"]
    client.download_file(
        Bucket="dreem-dod-h", Key=filename, Filename=path + "/dodh/{}".format(filename)
    )
