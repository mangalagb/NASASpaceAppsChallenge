import collections
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

# Read the CSA data set open data set
# It is the output of this url : https://www.asc-csa.gc.ca/eng/open-data/access-the-data.asp
def readHtml():
    base_path = Path(__file__).parent
    path = base_path / "../data/raw_data/html_files/Canada_open_data.html"

    data = ""
    with open(path, 'r') as f:
        data = f.read()
    return data

#Parse the html to extract the dataset's title and description.
#This is used as the training data for the recommender
def find_title_and_description(data):
    soup = BeautifulSoup(data, 'lxml')
    list_elements = soup.find_all('div', {"class": "panel panel-default"})
    summary_dict = collections.OrderedDict()

    for list_element in list_elements:
        #Extract the titles of the datasets
        title_div = list_element.find('div', {"class": "panel-heading"})
        title = extract_text_from_div(title_div)

        # Extract the summaries of the datasets
        summary_div = list_element.find('div', {"class": "field-description"})
        summary = extract_text_from_div(summary_div)

        summary_dict[title] = summary
    return summary_dict


def extract_text_from_div(text_div):
    text_span = str(text_div.find('span', {"class": "field-content"}))
    text = text_span.replace('<span class="field-content">', '')
    text = text.replace('</span>', '')
    text = re.sub('[^A-Za-z0-9 ]+', '', text)
    return text


def write_data_summaries_to_file(summary_dict):
    base_path = Path(__file__).parent
    path = base_path / "../data/raw_data/summaries/summary_file.txt"

    with open(path, 'w') as file:
        file.write(json.dumps(summary_dict))


def main():
    data = readHtml()
    summary_dict = find_title_and_description(data)
    write_data_summaries_to_file(summary_dict)

#Call main method
if __name__ == "__main__":
    main()
