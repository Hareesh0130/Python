import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from telegram import Bot
import time

# Telegram bot configuration
TELEGRAM_BOT_TOKEN = "7608908613:AAHuzK1fkMaZmNaT4qonM_LjiwGQ3b8tDww"
TELEGRAM_CHAT_ID = "1630000077"

# Function to scrape GMGN
def scrape_gmgn():
    print("Scraping GMGN...")
    url = "https://gmgn.com/trending-tokens"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tokens = []
    for card in soup.find_all("div", class_="token-card"):
        try:
            tokens.append({
                "name": card.find("h2").text.strip(),
                "volume": float(card.find("span", class_="volume").text.strip().replace("$", "").replace(",", "")),
                "liquidity": float(card.find("span", class_="liquidity").text.strip().replace("$", "").replace(",", "")),
                "age": int(card.find("span", class_="age").text.strip().replace(" hours", "")),
                "holders": int(card.find("span", class_="holders").text.strip().replace(",", "")),
                "contract_address": card.find("span", class_="contract-address").text.strip(),
            })
        except Exception as e:
            print(f"Error parsing token on GMGN: {e}")
    print(f"Scraped {len(tokens)} tokens from GMGN.")
    return tokens

# Function to scrape Dexscreener
def scrape_dexscreener():
    print("Scraping Dexscreener...")
    url = "https://dexscreener.com/trending"
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service("C:\\Users\\Hareesh\\Documents\\Python\\chromedriver_win32\\chromedriver.exe"), options=options)
    driver.get(url)
    time.sleep(5)
    tokens = []
    try:
        rows = driver.find_elements(By.CLASS_NAME, "trending-row")
        for row in rows:
            try:
                tokens.append({
                    "name": row.find_element(By.CLASS_NAME, "token-name").text.strip(),
                    "volume": float(row.find_element(By.CLASS_NAME, "token-volume").text.strip().replace("$", "").replace(",", "")),
                    "liquidity": float(row.find_element(By.CLASS_NAME, "token-liquidity").text.strip().replace("$", "").replace(",", "")),
                    "age": int(row.find_element(By.CLASS_NAME, "token-age").text.strip().replace(" hours", "")),
                    "holders": int(row.find_element(By.CLASS_NAME, "token-holders").text.strip().replace(",", "")),
                    "contract_address": row.find_element(By.CLASS_NAME, "token-ca").text.strip(),
                })
            except Exception as e:
                print(f"Error parsing token on Dexscreener: {e}")
    finally:
        driver.quit()
    print(f"Scraped {len(tokens)} tokens from Dexscreener.")
    return tokens

# Function to verify CA on RugCheck
def verify_on_rugcheck(contract_address):
    print(f"Verifying contract address on RugCheck: {contract_address}")
    url = "https://rugcheck.io"
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(service=Service("C:\\Users\\Hareesh\\Documents\\Python\\chromedriver_win32\\chromedriver.exe"), options=options)
    driver.get(url)
    time.sleep(3)
    try:
        driver.find_element(By.ID, "contract-address-input").send_keys(contract_address)
        driver.find_element(By.ID, "check-button").click()
        time.sleep(5)
        score = driver.find_element(By.CLASS_NAME, "safety-score").text.strip()
        print(f"RugCheck score: {score}")
        return score in ["Good", "Excellent"]
    except Exception as e:
        print(f"Error verifying on RugCheck: {e}")
        return False
    finally:
        driver.quit()

# Function to send message to Telegram bot
def send_to_telegram(message):
    print(f"Sending message to Telegram: {message}")
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# Main process
def main():
    print("Starting main process...")
    gmgn_tokens = scrape_gmgn()
    dexscreener_tokens = scrape_dexscreener()
    all_tokens = gmgn_tokens + dexscreener_tokens

    print(f"Total tokens to process: {len(all_tokens)}")
    for token in all_tokens:
        print(f"Checking token: {token['name']} with CA: {token['contract_address']}")
        if (token["liquidity"] < 100000 and token["volume"] < 250000 and 
            token["age"] >= 24 and token["holders"] <= 300):
            if verify_on_rugcheck(token["contract_address"]):
                message = (f"Token: {token['name']}\n"
                           f"Contract Address: {token['contract_address']}\n"
                           f"Liquidity: ${token['liquidity']:,}\n"
                           f"Volume: ${token['volume']:,}\n"
                           f"Age: {token['age']} hours\n"
                           f"Holders: {token['holders']}")
                send_to_telegram(message)
    print("Main process complete.")

if __name__ == "__main__":
    main()
