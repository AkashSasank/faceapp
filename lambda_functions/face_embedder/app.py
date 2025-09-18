import json

def handler(event, context):
    print(event, context)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "Test", "event": event})
    }
