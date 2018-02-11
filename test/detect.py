#!/usr/bin/env python

import requests

# put your keys in the header
headers = {
    "app_id": "76b48257",
    "app_key": "193cdafa1603d533b7cae71e81cb02df"
}

payload = '{"image":"https://media.kairos.com/liz.jpg"}'

url = "http://api.kairos.com/detect"

# make request
r = requests.post(url, data=payload, headers=headers)
print(r.content)