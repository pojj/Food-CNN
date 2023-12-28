import requests

resp = requests.post(
    "http://localhost:5000/predict",
    files={"image": open("data\\images\\shrimp_and_grits\\38604.jpg", "rb")},
)

print(resp.content)
