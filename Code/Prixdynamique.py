import requests

license_key = "ac8552236ca64d8ba76b6a20510e213e"  # Remplacez ceci par votre clé de licence

token = 'ac8552236ca64d8ba76b6a20510e213e' # from https://portal.realto.io
endpoint = 'https://api.realto.io/european-electricity-suppliers/list'


# call API and set-up DataFrame

response = requests.get(endpoint, headers={'OCP-Apim-Subscription-Key': token})
data = response.json()

print(data)