import os
import requests as req

# URL of the API to query
URL = "https://en.wikipedia.org/w/api.php"

# Selected medical categories
medical_categories = [
    'Category:Pathology',
    'Category:Pediatrics',
    'Category:Neurology',
    'Category:Cardiology',
    'Category:Oncology',
]

# Selected non-medical
other_categories = [
    'Category:Politics',
    'Category:Ecology',
    'Category:Electricity',
    'Category:Trigonometry',
    'Category:Artificial intelligence',
]


def get_text_from_page(pageid: str, is_medical: bool, test: bool):

    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "pageids": pageid,
        "formatversion": "2",
        "explaintext": 1
    }

    response = req.get(url=URL, params=params)
    data = response.json()

    text = data['query']['pages'][0]['extract']

    # Create directories
    if not os.path.exists('./corpus'):
        os.mkdir('./corpus')

        os.mkdir('./corpus/training')
        os.mkdir('./corpus/training/medical')
        os.mkdir('./corpus/training/non-medical')

        os.mkdir('./corpus/test')
        os.mkdir('./corpus/test/medical')
        os.mkdir('./corpus/test/non-medical')

    # Dividing documents into test and training set
    if test:
        if is_medical:
            with open(f'./corpus/test/medical/{pageid}.txt', 'w') as f:
                f.write(text)
        else:
            with open(f'./corpus/test/non-medical/{pageid}.txt', 'w') as f:
                f.write(text)
    else:
        if is_medical:
            with open(f'./corpus/training/medical/{pageid}.txt', 'w') as f:
                f.write(text)
        else:
            with open(f'./corpus/training/non-medical/{pageid}.txt', 'w') as f:
                f.write(text)


# Get the list of page ids given a specific wikipedia category
def get_documents(wiki_category: str, is_medical: bool):

    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "formatversion": "2",
        "cmtitle": wiki_category,
        "cmnamespace": "0",
        "cmtype": "page",
        "cmlimit": "100"
    }

    response = req.get(url=URL, params=params)
    data = response.json()

    # Number of found pages
    tot_pages = len(data["query"]["categorymembers"])

    print(f'For {wiki_category} found {tot_pages} documents.')

    try:
        for page in range(tot_pages):
            pageid = str(data["query"]["categorymembers"][page]['pageid'])

            # For each category: 80% training, 20% test
            if page < tot_pages * 4//5:
                get_text_from_page(pageid, is_medical, False)
            else:
                get_text_from_page(pageid, is_medical, True)
    except KeyError:
        print(f'Invalid category: {wiki_category}\n{data}')
        exit(-1)


def main():
    for mcat in medical_categories:
        get_documents(mcat, True)

    for ocat in other_categories:
        get_documents(ocat, False)


if __name__ == '__main__':
    main()
