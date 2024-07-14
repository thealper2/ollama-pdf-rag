import json
import requests
from config import Config

def pull_model(model_name, base_url):
    print(f"Pulling model: {model_name}")
    url = f"{base_url}/api/pull"
    data = json.duımps(dict(name=model_name))
    headers = {"Content-Type": "application/json"}

    with requests.post(url, data=data, headers=headers, stream=True) as response:
        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    print(chunk.decode('utf-8'), end='')

        else:
            print(f"Error: {response.status_code} - {response.text}")


model_name = Config.MODEL
base_url = Config.BASE_URL
pull_model(model_name, base_url)