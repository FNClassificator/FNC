import requests
import json
from bs4 import BeautifulSoup

from src.scripts.scrapper.resources.css_attributes import *
from src.utils import log
from src.utils import io

headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.106 Safari/537.36'
}

TYPE_TWO = [ELPLURAL]
TYPE_THREE = [HAYNOTICIAS]
TYPE_FOUR = [ELINFOBAE, PUBLICO]

def get_title(soup, attr):
    # Get title
    if 'attr' in attr['title']:
        title_soup = soup.find(attr['title']['name'], attr['title']['attr'])
    else:
        title_soup = soup.find(attr['title']['name'])
    title = title_soup.text
    return title.replace("\n", "")


def get_subtitle(soup, attr):
    # Get subtitle
    if 'subtitle' not in attr:
        return -1

    if 'attr' in attr['subtitle']:
        subtitle_soup = soup.find(
            attr['subtitle']['name'], attr['subtitle']['attr'])
    else:
        subtitle_soup = soup.find(attr['subtitle']['name'])
    if subtitle_soup != None:
        subtitle = subtitle_soup.text
        return subtitle.replace("\n", "")
    else:
        return ''


def get_body(soup, attr):
    if 'attr' in attr['text']:
        body_soup = soup.find(attr['text']['name'], attr['text']['attr'])
    else:
        body_soup = soup.find(attr['text']['name'])
    
    paragraph_list = []
    for paragraph_soup in body_soup.find_all('p', recursive=False):
        text_aux = paragraph_soup.text
        if text_aux:
            text_aux = text_aux.replace("\n", "")
            if len(text_aux) > 2:  # Check if is empty
                paragraph_list.append(text_aux)
    return paragraph_list

def get_body_type_two(soup,attr):
    body_soup = soup.find(attr['text']['name'], attr['text']['attr'])
    paragraph_list = str(body_soup).split('<br/>')
    result = []
    for p in paragraph_list:
        clean_p = BeautifulSoup(p)
        if len(clean_p.text) > 2:
            result.append(clean_p.text)

    return result

def get_body_type_three(soup,attr):
    body_soup = soup.find(attr['text']['name'], attr['text']['attr'])
    
    paragraph_list = []

    # Especial first paragraph (optional)
    first_p = body_soup.find('p')
    if first_p != None and len(first_p.text) > 2:
        paragraph_list.append(first_p.text)
    # The others ...
    first_child = body_soup.find('div')
    for paragraph_soup in first_child.find_all('p', recursive=False):
        text_aux = paragraph_soup.text
        if text_aux:
            text_aux = text_aux.replace("\n", "")
            if len(text_aux) > 2:  # Check if is empty
                paragraph_list.append(text_aux)
    return paragraph_list

def get_body_type_four(soup,attr):
    body_soup = soup.find_all(attr['text']['name'], attr['text']['attr'])
    paragraph_list = []
    for paragraph_soup in body_soup:
        text_aux = paragraph_soup.text
        if text_aux:
            text_aux = text_aux.replace("\n", "")
            if len(text_aux) > 2:  # Check if is empty
                paragraph_list.append(text_aux)
    return paragraph_list


def make_request(url, attr):
    response = requests.get(url, headers=headers)
    result = {}
    if response.status_code == 200:
        data = response.text
        soup = BeautifulSoup(data, 'html.parser')

        # Title
        result['title'] = get_title(soup, attr)
        # Subtitle
        subtitle = get_subtitle(soup, attr)
        if subtitle != -1:
            result['subtitle'] = subtitle
        else:
            result['subtitle'] = ''
        # Text
        if attr in TYPE_TWO:
            result['text'] = get_body_type_two(soup, attr)
        elif attr in TYPE_THREE:
            result['text'] = get_body_type_three(soup, attr)
        elif attr in TYPE_FOUR:
            result['text'] = get_body_type_four(soup, attr)
        else:
            result['text'] = get_body(soup, attr)
        # URL
        result['url'] = url
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
        'periodistadigital.com': PERIODISTADIGITAL,
        'elplural.com': ELPLURAL,
        'antena3.com': ANTENA3,
        'lasexta.com': LASEXTA,
        'elperiodico.com': ELPERIODICO,
        'elmundo.es': ELMUNDO,
        '20minutos.com': XXMINUTOS,
        'elpais.com': ELPAIS,
        'gaceta.es': LAGACETA,
        # 'elespanol.com': ELESPANOL
        'haynoticias.es' : HAYNOTICIAS,
        'ecoportal.net': ECOPORTAL,
        'europafm.com': EUROPAFM,
        'elconfidencialdigital.com': ELCONFIDENCIAL,
        'elinfobae.com': ELINFOBAE,
        'eleconomista.es': ELECONOMISTA,
        'ultimahora.com': ULTIMAHORA,
        'cadenaser.com': CADENASER,
        'publico.es': PUBLICO,
        'que.es': QUE,
        'fcinco.com': FCINCO
    }
    return switcher.get(var, -1)


def get_all_news(path, is_fake, n):
    url_list = io.read_json_file(path)
    for key in url_list.keys():
        css_attr = get_variable(key)
        if css_attr != -1:
            for item in url_list[key]:
                n += 1
                try:
                    print('Making request of ...', item)
                    result = make_request(item, css_attr)
                    result['fake'] = is_fake
                    out_path = 'src/data/articles/Article_' + str(n) + '.json'
                    io.write_json_file(out_path, result)
                    print('File saved!')
                except ValueError as err:
                    print(err)
                    print('Problem with: ', item)
    return n


def run(n):
    path = 'src/data/list_url_2.json'
    n = get_all_news(path, False, n)
    print(n)


if __name__ == '__main__':
    # From
    n = 63
    run(n)
