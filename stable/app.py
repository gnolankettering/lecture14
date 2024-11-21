import requests
import config 

api_key=config.API_KEY

response = requests.post(
    f"https://api.stability.ai/v2beta/stable-image/generate/core",
    headers={
        "authorization": f"Bearer {api_key}",
        "accept": "image/*"
    },
    files={"none": ''},
    data={
        "prompt": "Lighthouse on a cliff overlooking the ocean",
        "output_format": "webp",
    },
)

if response.status_code == 200:
    with open("./lighthouse.webp", 'wb') as file:
        file.write(response.content)
else:
    raise Exception(str(response.json()))