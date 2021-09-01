import requests, uuid, json

# Add your subscription key and endpoint
subscription_key = "53a2d9aeb234442d912203e1e315ae06"
endpoint = "https://api.cognitive.microsofttranslator.com"

# Add your location, also known as region. The default is global.
# This is required if using a Cognitive Services resource.
# location = "eastus"

path = '/translate'

constructed_url = endpoint + path
headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    # 'Ocp-Apim-Subscription-Region': location,
    'Content-type': 'application/json',
    'X-ClientTraceId': str(uuid.uuid4())
}

def trans_bing(string, src_lang, dst_lang):
    
    params = {
    'api-version': '3.0',
    'from': src_lang,
    'to' : dst_lang
    # 'to': ['de', 'it']
    }
    
    # You can pass more than one object in body.
    body = [{
        'text': string
    }]

    request = requests.post(constructed_url, params=params, headers=headers, json=body)
    response = request.json()

    # print(json.dumps(response, sort_keys=True, ensure_ascii=False, indent=4, separators=(',', ': ')))
    return response

