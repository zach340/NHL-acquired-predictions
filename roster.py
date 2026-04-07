import requests
url = "https://api-web.nhle.com/v1/roster/EDM/20252026"
resp = requests.get(url, timeout=15)
data = resp.json()
print(list(data.keys()))
print(data["forwards"][0])