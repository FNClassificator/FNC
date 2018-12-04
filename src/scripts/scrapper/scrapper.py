import requests
import json
from bs4 import BeautifulSoup

from src.scripts.scrapper.resources.css_attributes import *
from src.utils import log
from src.fake_news_detector.helpers.read_data import io

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.106 Safari/537.36'
}


def make_request(url, attr):
    response = requests.get(url, headers=headers)
    result = {}
    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, 'html.parser')

        # Get title
        if 'attr' in attr['title']:
            title_soup = soup.find(attr['title']['name'], attr['title']['attr'])
        else:
            title_soup = soup.find(attr['title']['name'])
        title = title_soup.text
        result['title'] = title.replace("\n", "")
 
        # Get subtitle
        if 'attr' in attr['subtitle']:
            subtitle_soup = soup.find(attr['subtitle']['name'], attr['subtitle']['attr'])
        else:
            subtitle_soup = soup.find(attr['subtitle']['name'])
        subtitle = subtitle_soup.text
        result['subtitle'] = subtitle.replace("\n", "")

        # Get text
        body_soup = soup.find(attr['text']['name'], attr['text']['attr'])
        paragraph_list = []
        for paragraph_soup in body_soup.find_all('p', recursive=False):
            text_aux = paragraph_soup.text
            if text_aux:
                text_aux = text_aux.replace("\n", "")
                if len(text_aux) > 2:
                    paragraph_list.append(text_aux)
                else:
                    pass
        result['text'] = paragraph_list
        return result


def get_variable(var):
    switcher = {
        'abc.com': ABC,
        'lavanguardia.com': LA_VANGUARDIA,
        'marca.com': MARCA,
        'esdiario.com': ESDIARIO,
        'mediterraneodigital.com': MEDITERRANEO,
        'alertadigital.com': ALERTADIGITAL,
        'cataladigital.com': CATALADIGITAL,
        'europapress.com': EUROPAPRESS,
        'gomeraactualidad.com': GOMERAACTUALIDAD,
        'periodistadigital.com': PERIODISTADIGITAL
    }
    return switcher.get(var, -1)

def read_json_file(path):
    try:
        log.info('Reading ' + path + ' file ...')
        with open(path, 'r') as f:
            log.info(path + ' file opened.')
            return json.load(f)
    except IOError as err:  # whatever reader errors you care about
        log.error('IOError - File not found or couldn\'t be open')
        log.error(err)

def get_all_news(path, is_fake, n):
    url_list = read_json_file(path)
    url_list = {
        "cuatro.com": [
            "https://www.cuatro.com/deportes/croacia-donara-dinero-mundial-causas-beneficas_0_2596650092.html"
        ]
    }
    for key in url_list.keys():
        css_attr = get_variable(key)
        if css_attr != -1:
            for item in url_list[key]:
                n += 1
                result = make_request(item, css_attr)
                result['fake'] = is_fake
                out_path = 'src/data/articles/Article_' + str(n) + '.json'
                io.write_json_file(out_path, result)
    return n


def run(n):
    path = 'src/data/list_url.json'
    n = get_all_news(path,True,n)
    print(n)


if __name__ == '__main__':
    #From 
    n = 99
    run(n)
