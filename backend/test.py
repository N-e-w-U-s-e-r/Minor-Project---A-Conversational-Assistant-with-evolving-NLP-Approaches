import requests

request = requests.post(
    "http://127.0.0.1:5000/api/chat",
    json = {
        "message": "hello, flask"
    }
)

print(request.status_code)
print(request.headers)
print(request.text)
print(request.json())