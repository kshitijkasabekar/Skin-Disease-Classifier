from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time

# Set up Chrome WebDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

# Launch Streamlit app in a browser
driver.get("http://localhost:8501")  # Replace with the URL of your Streamlit app

# Upload image
upload_button = driver.find_element("xpath", "//input[@type='file']")
upload_button.send_keys("03DermatitisArm1.jpg")  # Replace with the path to your test image

# Click on the classify button
classify_button = driver.find_element("xpath", "//button[contains(text(), 'Classify')]")
classify_button.click()

# Wait for classification result
time.sleep(5)

# Verify the result
result_element = driver.find_element("xpath", "//p[contains(text(), 'Prediction')]")
prediction = result_element.text
assert prediction == "Prediction: Your_predicted_class", "Prediction does not match expected result"

# Close the browser
driver.quit()

