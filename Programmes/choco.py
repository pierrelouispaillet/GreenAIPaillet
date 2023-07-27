import requests
from bs4 import BeautifulSoup

url = "https://fr.openfoodfacts.org/produit/7622210869616/chocolat-bio-noir-85-cote-d-or"
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

# Carbon footprint info may not be in a table format, 
# or its table might not have the same 'aria-label' as the nutritional information.
# You would need to inspect the HTML of the webpage to find out how to scrape it.
