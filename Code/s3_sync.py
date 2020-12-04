#utilities for syncing to/from s3 file
import boto3
import tempfile
import os 
global to_clean
#delete tmp files at the end of the run
to_clean=[]

def s3_string_parse(s3_string):
    #sync log, model hdf5, weight file, arch file
    s3_client=boto3.client(service_name='s3')
    bucket=s3_string.lstrip('s3://').split('/')[0]
    filename="/".join(s3_string.lstrip("s3://").split("/")[1::])
    return bucket,filename

#target file for local download is either provided or inferred as the basenme
# of the s3 file path 
def download_s3_file(s3_source,s3_target=None):
    s3 = boto3.client('s3')
    bucket,s3_file=s3_string_parse(s3_source)
    if s3_target is None:
        s3_target=s3_file.split('/')[-1]
    s3.download_file(bucket, s3_file, s3_target)
    print("download of "+s3_source+ " to " + s3_target + " is complete")
    to_clean.append(s3_target)
    return s3_target

def upload_s3_file(s3_target,source_filename):
    s3 = boto3.client('s3')
    bucket,s3_file=s3_string_parse(s3_target)
    s3.upload_file(source_filename,bucket,s3_file)
    to_clean.append(source_filename)
    print("upload of "+s3_target+" is complete")


def read_s3_file_contents(s3_string):
    s3=boto3.resource('s3')
    bucket_name,itemname=s3_string_parse(s3_string)
    obj=s3.Object(bucket_name,itemname)
    contents=obj.get()['Body'].read().decode('utf-8').strip()
    return contents
    

def run_cleanup():
    to_clean_set=list(set(to_clean))
    print("to_clean_set:"+str(to_clean_set))
    if len(to_clean_set)>0:
        for f in to_clean_set:
            try:
                os.remove(f)
                print("deleted local file:"+str(f))
            except:
                continue
    return 
        
