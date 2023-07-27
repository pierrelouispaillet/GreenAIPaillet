import requests
from bs4 import BeautifulSoup

url = 'https://fr.openfoodfacts.org/produit/7622210869616/chocolat-bio-noir-85-cote-d-or'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

def scrape_table(table_aria_label):
    table = soup.find('table', attrs={'aria-label': table_aria_label})
    headers = [header.text.strip() for header in table.find_all('th')]
    rows = table.find_all('tr')
    table_data = []
    for row in rows[1:]:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        table_data.append({headers[i]: cols[i] for i in range(len(cols))})
    return table_data

# Scrape nutritional information
nutritional_info = scrape_table('Tableau nutritionnel')
print("Nutritional Information:")
for info in nutritional_info:
    print(info)

# Search for the panel that contains the carbon footprint data
carbon_footprint_panel = soup.find('ul', {'id': 'panel_carbon_footprint'})

# Within this panel, find the h5 element that contains the specific information
carbon_footprint_info = carbon_footprint_panel.find('h5').text

print("\nCarbon Footprint Information:")
print(carbon_footprint_info)
