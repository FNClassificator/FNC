
import json
from src.utils import io

articles = {
    'articles' : []
}
n_ini = 1
n_fi = 116
for x in range(n_ini,n_fi):
    path =  'src/data/articles_en/Article_' + str(x) + '.json'
    # Read file
    content = io.read_json_file(path)
    if content != None:
        articles['articles'].append(content)
io.write_json_file('src/data/tmp.json', articles)