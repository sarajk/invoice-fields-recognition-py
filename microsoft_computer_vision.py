from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import json

mcv_configuration_path = 'input/mcv_configuration.json'

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = ""
endpoint = ""

if not os.path.exists(mcv_configuration_path):
    raise RuntimeError(mcv_configuration_path + ' file is missing!')
else:
    with open(mcv_configuration_path) as f:
        config = json.load(f)
        subscription_key = config['subscriptionKey']
        endpoint = config['endpoint']

if len(subscription_key.strip()) == 0 or len(endpoint.strip()) == 0:
    raise RuntimeError('Microsoft Computer Vision Subscription key or Endpoint is missing, make sure that you have an mcv_configuration.json created!')


computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

print('Initialized Microsoft Computer Vision.')

'''
OCR: Read File using the Read API, extract text - local
This example will extract text in an image, then print results, line by line.
This API call can also extract handwriting style text (not shown).
'''
def analyze_image(image_file_path):

    # Get an image with text
    image_file = open(image_file_path, 'rb')
    # Call API with URL and raw response (allows you to get the operation location)
    read_response = computervision_client.read_in_stream(image_file, raw=True)

    # Get the operation location (URL with an ID at the end) from the response
    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results 
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        return read_result.analyze_result
    return None