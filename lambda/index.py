import json
import os

import boto3

RUNTIME_CLIENT = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = "mobilebert-endpoint"


def handler(event, context):
    payload = event['body']
    json_payload = json.dumps(payload)

    response = RUNTIME_CLIENT.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=json_payload
    )

    result = response['Body'].read().decode()

    return {
        'statusCode': 200,
        'body': json.dumps({'result': result})
    }
