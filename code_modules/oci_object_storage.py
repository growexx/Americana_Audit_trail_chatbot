"""
Module to interact with OCI Object Storage buckets.

This module provides functionality to:
- Download files from an OCI Object Storage bucket
- Upload files to an OCI Object Storage bucket
"""
import oci
import configparser
import os
import logging

# Configure application-level logging
logger = logging.getLogger(__name__)


class OCIObjectStorageClient(object):
    """
    Simple Class for interacting with OCI Object Storage Bucket
    """
    def __init__(self):
        """Initialize OCI client by reading config

        Initializes and returns an OCI Object Storage client and bucket information
        by reading configuration from `config.ini`.

        Returns:
            tuple: A tuple containing:
                - oci.object_storage.ObjectStorageClient: The initialized OCI Object Storage client.
                - dict: A dictionary containing bucket namespace, bucket name, compartment ID, and region.
        """
        config = configparser.ConfigParser()
        config.read('config.ini')
        # OCI authentication configuration
        oci_config = {
            "user": config['DEFAULT']['user'],
            "key_file": config['DEFAULT']['key_file'],
            "fingerprint": config['DEFAULT']['fingerprint'],
            "tenancy": config['DEFAULT']['tenancy'],
            "region": "me-dubai-1"
        }

        # Initialize OCI Object Storage client
        self.bucket_info = {
            "namespace": "bmb8tbvmgtsy",
            "bucket_name": "bucket-convbirealestate",
            "compartment_id": "ocid1.compartment.oc1..aaaaaaaa7wyo7euk2wfekpv36obtfbqgupxeb5yylivifscxseudvwwp2ixa",
            "region": "me-dubai-1"
        }

        self.client = oci.object_storage.ObjectStorageClient(oci_config)

    def get_pdf_files_from_bucket(self,filename):
        """Get all PDF file names and URLs from pdf folder in bucket

        Retrieves a list of PDF file names and their corresponding URLs from the 'pdf' folder
        within the configured OCI Object Storage bucket.

        Returns:
            tuple: A tuple containing:
                - list: A list of cleaned PDF file names (without 'pdf/' prefix or '.pdf' extension).
                - list: A list of URLs for the original PDF objects in the bucket.
        """
        logger.info("Get file names from bucket")

        namespace = self.bucket_info['namespace']
        bucket_name = self.bucket_info['bucket_name']


        # List objects in pdf folder
        response = self.client.get_object(
            namespace_name=namespace,
            bucket_name=bucket_name,
            object_name=filename
        )
        download_path = filename
        with open(download_path, "wb") as f:
            for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

        print("Download completed:", download_path)

        return download_path

    def upload_file_to_bucket(self,file_name,bucket_folder_name):
        """Upload file to csv folder in bucket

        Uploads a specified local file to the 'csv-ps' folder within the configured
        OCI Object Storage bucket. The uploaded file is given a 'text/csv' content type.

        Args:
            file_name (str): The path to the local file to be uploaded.

        Returns:
            str: The object name (path) of the uploaded file in the bucket.
        """
        logger.info(f"uploading file {file_name} to bucket ")

        # Read file from local directory
        with open(file_name, 'rb') as f:
            file_content = f.read()

        # Get just the file_name from path
        object_name = bucket_folder_name + file_name.split("/")[-1]

        response = self.client.put_object(
            namespace_name=self.bucket_info['namespace'],
            bucket_name=self.bucket_info['bucket_name'],
            object_name=object_name,
            put_object_body=file_content,
            content_type="text/csv"
        )
        logger.info(f"Uploading process finished {response}")

        return object_name
