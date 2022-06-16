import requests 
from selectorlib import Extractor

# Get data from an amazon url
def scrape(url):  
    e = Extractor.from_yaml_file('template.yml')
    headers = {
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://www.amazon.com/',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    # Download the page using requests
    print("Downloading %s"%url)
    r = requests.get(url, headers=headers)
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
        return None
    # Pass the HTML of the page and create 
    return e.extract(r.text)

# Function to process results
def extract_data(url):
    data = scrape(url)
    new_data = {}
    new_data['image'] = data['Image']
    new_data['Attribute'] = []
    new_data['Value'] = []
    new_data['title'] = data['Info']['Title']
    for i in range(1, 10):
        attr = 'attr' + str(i)
        val = 'val' + str(i)
        if data['Info'][attr] != None:
            new_data['Attribute'].append(data['Info'][attr])
            new_data['Value'].append(data['Info'][val])
    return new_data