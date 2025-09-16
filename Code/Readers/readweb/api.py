import requests

def readAPI(url, params, headers):  
    response = requests.get(url, params=params, headers=headers).json()
    return response