from locust import HttpUser, between, task
import os

class SkinDiseaseClassifierUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def classify_image(self):
        # Path to the test image
        image_path = "03DermatitisArm1.jpg"
        
        # Send a POST request to upload the image
        with open(image_path, "rb") as image_file:
            files = {'file': image_file}
            response = self.client.post("/upload", files=files)
        
        # Ensure the image was uploaded successfully
        if response.status_code != 200:
            self.environment.runner.quit()
        
        # Send a GET request to trigger classification
        response = self.client.get("/classify")
        
        # Ensure the classification request was successful
        assert response.status_code == 200

    def on_start(self):
        # Set base URL for the Streamlit app
        self.client.base_url = os.getenv("STREAMLIT_BASE_URL", "http://localhost:8501")
