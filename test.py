from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
driver = webdriver.Chrome(service=Service("C:\\Users\\Hareesh\\Documents\\Python\\chromedriver_win32\\chromedriver.exe"), options=options)

driver.get("https://www.google.com")
print(driver.title)  # Should print "Google"
driver.quit()
