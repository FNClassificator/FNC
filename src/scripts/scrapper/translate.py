import os
from src.fake_news_detector.helpers.read_data import io
from src.fake_news_detector.helpers.translate import translate


def request_title_subtitle(content):
    if content['title'] == content['subtitle']:
        resp = translate.make_request([content['title']])
        title = resp['translations'][0]['text']
        subtitle = title
    else:
        resp = translate.make_request([content['title'], content['subtitle']])
        title = resp['translations'][0]['text']
        subtitle = resp['translations'][0]['text']
    return title, subtitle

def request_body(content):
    rest_two = translate.make_request(content['text'])
    text_body = []
    for elem in rest_two['translation']:
        text_body.append(elem['text'])
    return text_body


def run(nmin,nmax):
    
    for x in range(nmin, nmax):
        translate = { }
        path = 'src/data/articles/Article_' + x + '.json'
        print('Reading file ...', path)
        content = io.read_json_file(path)
        print('Done!')

        print('Requesting title and subtitle tranlation...')
        # Title and subtitle request
        resp_one = request_title_subtitle(content)
        translate['title'] = resp_one[0]
        translate['subtitle'] = resp_one[1]
        print('Requesting text body tranlation...')
        # Text body request
        rest_two = request_body(content)
        translate['text'] = rest_two
        print('Done!')

        print('Writting article in English ...')
        path_dest = 'src/data/articles_en/Article_' + x + '.json'
        io.write_json_file(path_dest, translate)
        print('Done!')