from locust import HttpUser, task, between

with open("../samples/ant.jpeg", "rb") as f:
    test_image_bytes = f.read()


class PyTorchMNISTLoadTestUser(HttpUser):

    wait_time = between(0.01, 2)

    @task
    def predict_image(self):
        files = {"upload_files": ("ant.jpeg", test_image_bytes, "image/jpeg")}
        self.client.post("/predict_image", files=files)