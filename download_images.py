#!/usr/bin/python

import boto3
import os
import json
import sys
from os.path import isfile, join, basename, exists

def download_s3_file(ddir, s3, bucket_name, fkey):
    fname = join(ddir, basename(fkey))
    if not exists(fname):
        s3.download_file(bucket_name, fkey, fname)
        print("File " + fname + " is downloaded")
    else:
        print("File " + fname + " is already downloaded")

def download_samples(ddir):
    if not exists(ddir):
        os.makedirs(ddir)
        
    bucket_name = "smile-detection-test"
    s3 = boto3.client('s3', region_name='eu-west-1')
    objs = {}
    with open("image_to_smile.json", "r") as fh:
        objs = json.load(fh).keys()
        
    for f in objs:
        download_s3_file(ddir, s3, bucket_name, f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        download_samples(sys.argv[1])
    else:
        print("Usage: download_images.py dirname")
