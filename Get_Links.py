from splinter import Browser
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

# Load the CSV file
csv_file = r'C:\Users\adabi\OneDrive\Documents\NFL_Contract_Evals\drafted_wrs_combined.csv'
data = pd.read_csv(csv_file)
data = data[data['Year'] <=2025]
# Initialize an empty list to store player data
player_data = []

# Set up the browser using ChromeDriverManager
service = Service(ChromeDriverManager().install())
with Browser('chrome', service=service) as browser:
    # Iterate through the links and extract required sections
    for index, row in data.iterrows():
        link = row[0]
        player_name = row[1]
        college = row[2]
        pick = row[3]
        position = row[4]
        team = row[5]
        year = row[6]
        
        player_text = {}
        try:
            # Visit the link using the browser
            browser.visit(link)
            time.sleep(2)  # Wait for the page to load
            # Find and click the 'Accept Cookies' button if it exists
            try:
                if browser.is_element_present_by_text('Accept Cookies', wait_time=5):
                    browser.find_by_text('Accept Cookies').first.click()
            except Exception as e:
                print(f"Error clicking 'Accept Cookies' button: {e}")
            time.sleep(2)  # Wait for the page to load after clicking
            # Check if the page has loaded correctly
            # if browser.is_text_present('Player Bio', wait_time=5):
            #     print(f"Page loaded successfully for {player_name}")
            # else:
            #     print(f"Page did not load correctly for {player_name}")
                # continue  # Skip to the next link if the page didn't load
            # Scrape the entire webpage
            html = browser.html
            soup = BeautifulSoup(html, 'html.parser')

            # Extract the player-bio section
            player_bio = soup.find('div', class_='css-1gm5rd1-Af')

            # Extract the analysis section
            analysis_section = soup.find_all('div', class_='css-11b5sq2-Af')
            # Extract the first analysis section
            analysis = analysis_section[0].text if analysis_section else None
            strengths = analysis_section[1].text if len(analysis_section) > 1 else None
            weaknesses = analysis_section[2].text if len(analysis_section) > 2 else None
            player_text= {
                'Player Name': player_name,
                'Link': link,
                'Analysis': analysis,
                'Strengths': strengths,
                'Weaknesses': weaknesses,
                'College': college,
                'Pick': pick,
                'Position': position,
                'Team': team,
                'Year': year
            }
            player_data.append(player_text)
            html = browser.html
            soup = BeautifulSoup(html, 'html.parser')
            
            # # Print all items found on the page for troubleshooting

        except Exception as e:
            print(f"Error processing {player_name}: {e}")
# Convert player_data into a DataFrame
player_df = pd.DataFrame(player_data)

# Read the existing CSV file if it exists
output_csv_file = r'C:\Users\adabi\OneDrive\Documents\NFL_Contract_Evals\WR_data.csv'
try:
    existing_data = pd.read_csv(output_csv_file)
    # Append the new data to the existing data
    combined_data = pd.concat([existing_data, player_df], ignore_index=True)
except FileNotFoundError:
    # If the file doesn't exist, use the new data as the combined data
    combined_data = player_df

# Save the combined DataFrame back to the CSV file
combined_data.to_csv(output_csv_file, index=False)

print(f"Player data saved to {output_csv_file}")
print("Finished processing all links.")
print(player_df.head(1))