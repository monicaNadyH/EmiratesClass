import pandas as pd
import json
from watson_developer_cloud.natural_language_understanding_v1 import Features, SentimentOptions
from watson_developer_cloud import NaturalLanguageClassifierV1, NaturalLanguageUnderstandingV1
from tqdm import tqdm

tqdm.pandas()

print('initializing...')

primary = '6876e8x557-nlc-1148'
secondary = {
    'Cabin Crew': '6874ebx556-nlc-1066',
    'Airport': '6876e8x557-nlc-1146',
    'Medical': '687c74x560-nlc-1053',
    'Duty Free': '68788bx558-nlc-1120',
    'Security': '687c74x560-nlc-1055',
    'Safety': '68788bx558-nlc-1123',
    'Emirates Skywards': '6f1d8ax561-nlc-950',
    'Product Development': '6f1d8ax561-nlc-951'
}

nlu = NaturalLanguageUnderstandingV1(
    version='2017-02-27',
    username="6665231d-1726-49fa-9fad-32bdbfcc7e5f",
    password='ktIRKMEozS1x')

nlc_p = NaturalLanguageClassifierV1(
    username="1ae5c142-c969-42ea-b8f8-934f1855b6bc",
    password='iniUKzRb8AhD')

nlc_s = NaturalLanguageClassifierV1(
    username="9b619a5a-c842-4a95-b4c4-0a5cdb99f326",
    password='yBSmyX0ArMuw')

print('reading...')
data = pd.read_csv('testv1.csv')

def sentiment(text):
    try:
        response = nlu.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions()))
        
        sentiment = json.dumps(response['sentiment']['document']['label'])
    except Exception as e:
        print(e)
        sentiment = "No Enough Text"

    return sentiment.strip('"')

def classifyP(text):

    classes = nlc_p.classify(primary, text)
    cl = json.dumps(classes['classes'][0]['class_name'])
    conf = json.dumps(classes['classes'][0]['confidence'])
    
    try:
        if float(conf) < 0.75:
            conf = "Please manually Review"
    except Exception as e:
        print(cl, e)

    return cl.strip('"'), conf

def classifySub(text, sub):

    classes = nlc_s.classify(secondary[sub], text)
    cl = json.dumps(classes['classes'][0]['class_name'])    
    conf = json.dumps(classes['classes'][0]['confidence'])
    
    try:
        if float(conf) < 0.75:
            conf = "Please manually Review"
    except Exception as e:
        print(cl, e)

    return cl.strip('"'), conf


def clean(text):
    t = str(text).replace('=', '').replace('#', '').replace('*', ' ').replace('-', ' ').replace('+', '').replace('_', '').replace('<', '').replace('>', '').replace('[', '').replace(']', '').replace('{', '').replace('}', '').replace('/', '').replace('|', '').replace('.', '').replace('$', '').replace('"', '')
    " ".join(t.split())
    return t.strip()

def process(row):
    row['description'] = clean(row['description'])
    row['sentiment'] = sentiment(row['description'])
    row['PClass'], row['PConfidence'] = classifyP(row['description'])
    row['SClass'], row['SConfidence'] = classifySub(row['description'], row['PClass'])
    return row

print('Processing...')
data = data.progress_apply(lambda row: process(row), axis=1)

print('saving...')
data.to_csv('results.csv', index=False)
