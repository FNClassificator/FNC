# FNC

## Requisites

1. Python +3.5
2. Virtualenv
3. Java (to execute classificators)

## Set up
To create a new envioment for the project

1. Set up the virtualenvioment:
```
virtualenv env --python=python3.X
```
`NOTE`: on python3.X, insert your current version of python.

2. Activate the enviorment:
```
source env/bin/activate
```

3. Install requeriments:
```
pip3 install -r src/requeriments.txt
```
4. Enjoy!
Now you can run all files, except which uses Standford parser. Theses files needs to be executed at the same time a parallel server in Java to use this dependence. The set up is going to be explain in the following section.


## Dependences

[IMPORTANT!] To execute classificator it needs to excecute in parallel the Standord parser in java enviorment. Follow the instructions to get and excecute it.
1. Download the Standford Parser

(Important) Open a new terminal and excecute the following line inside root folder:
```
mkdir models & cd models
```

Inside models folder donwload the package:
- **OPTION 1**: (click here)[https://stanfordnlp.github.io/CoreNLP/download.html]
- **OPTION 2**: with wget

```
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
```

1. Unzip the file

```
unzip stanford-corenlp-full-2018-10-05.zip
```

3. Excecute the program in the current terminal

```
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
```

## Usage

### Folders

1. **data**

Here there are the articles of dataset, in Spanish and English.
First the file `list_url.json` includes all fake news's urls that where scrapped into the folder `articles/`, and then translated to `artciles_en/`.

2. **fake_news_detector**
3. **fake_news_detector/runners**
4. **scripts**