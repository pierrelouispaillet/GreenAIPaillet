import requests
import csv

# Define the base url
base_url = 'https://world.openfoodfacts.org/cgi/search.pl'

# Define the parameters for the API request
params = {
    'search_terms': 'tablette chocolat',
    'action': 'process',
    'json': 1,  # We want the response to be in JSON format
    'page_size': 1000,
}

# Send the GET request
response = requests.get(base_url, params=params)

# Parse the JSON response
data = response.json()

# Extract the products
products = data['products']

# Write the data to a CSV file
with open('products.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['product_name', 'quantity', 'nutrition_score', 'carbon_footprint', 'co2_packaging', 'store_count', 'country count', 'ingredients'])
    # Write the product data
    for product in products:
        product_name = product.get('product_name', 'N/A')
        quantity = product.get('quantity', 'N/A')
        nutrition_score = product.get('nutriscore_data', {}).get('score', 'N/A')
        carbon_footprint = product.get('ecoscore_data', {}).get('agribalyse', {}).get('co2_total', 'N/A')
        co2_packaging = product.get('ecoscore_data', {}).get('agribalyse', {}).get('co2_packaging', 'N/A')
        ingredients_text = product.get('ingredients_text', 'N/A')
        stores = product.get('stores', '')
        store_count = len(stores.split(',')) if stores else 0
        countries = product.get('countries', '')
        countries_count = len(countries.split(',')) if countries else 0


        writer.writerow([product_name, quantity, nutrition_score, carbon_footprint, co2_packaging, store_count , countries_count ,ingredients_text])
