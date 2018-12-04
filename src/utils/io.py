from src.utils import log
import json

# Read JSON file
# Pre: JSON file's *path*
# Post: Content of JSON path
def read_json_file(path):
    try:
        log.info('Reading ' + path + ' file ...')
        with open(path, 'r') as f:
            log.info(path + ' file opened.')
            return json.load(f)
    except IOError as err: # whatever reader errors you care about
        log.error('|IOError| - File not found or couldn\'t be open')

# Write dictionary or JSON in JSON file
# Pre: *path* where file is going to be store, *content* dictionary's content
# Post: File in path *path* is gonna be created.
def write_json_file(path, content):
    try:
        log.info('Opening ' + path + ' file ...')
        with open(path, 'w') as outfile:
            json.dump(content, outfile,  ensure_ascii=False)
            log.info(path + ' file writed.')
    except IOError as err: # whatever reader errors you care about
        log.error('|IOError| - File not found or couldn\'t be open to write')