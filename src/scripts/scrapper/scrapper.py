import requests

from bs4 import BeautifulSoup
from src.scripts.css_attributes import *

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.106 Safari/537.36'}


def make_request(url, attr):
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, 'html.parser')

        # Get title
        title_soup = soup.find(attr['title']['name'], attr['title']['attr'])
        title = title_soup.text
        print('Title:', title)

        # Get subtitle
        subtitle_soup = soup.find(
            attr['subtitle']['name'], attr['subtitle']['attr'])
        subtitle = subtitle_soup.text
        print('Subitle:', subtitle)

        # Get text
        body_soup = soup.find(attr['text']['name'], attr['text']['attr'])
        paragraph_list = []
        for paragraph_soup in body_soup.find_all('p', recursive=False):
            text_aux = paragraph_soup.text
            if text_aux:
                print('TEXT_AUX:', text_aux)
                paragraph_list.append(text_aux)
        print('TOTAL:', len(paragraph_list))
        return title, subtitle, paragraph_list


def get_variable(var):
    switcher = {
        'abc.com': ABC,
        'lavanguardia.com': LA_VANGUARDIA,
        'marca.com': MARCA
    }
    return switcher.get(var, "Invalid")


def read_json_file(path):
    try:
        log.info('Reading ' + path + ' file ...')
        with open(path, 'r') as f:
            log.info(path + ' file opened.')
            return json.load(f)
    except IOError as err:  # whatever reader errors you care about
        log.error('IOError - File not found or couldn\'t be open')


def run():
    path = 'app/data/list_url.json'
    url_list = read_json_file(path)
    for key in url_list['articles']:
        css_attr = get_variable(key)
        for item in url_list['articles'][key]:
            make_request(item, css_attr)


if __name__ == '__main__':
    url = 'https://www.abc.es/internacional/abci-madeleine-albright-esta-dispuesta-convertirse-islam-protesta-contra-trump-201701270955_noticia.html'
    make_request(url)
