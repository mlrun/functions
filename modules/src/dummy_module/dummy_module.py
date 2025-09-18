import requests

print("dummy module is being executed!")

def greet(name):
    return f"Hello, {name}!"

def fetch_httpbin():
    """Make a test GET request to httpbin.org"""
    response = requests.get("https://httpbin.org/get")
    return response.json()

VALUE = 42