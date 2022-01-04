import requests

def conceptnet(word):
    try:
        obj = requests.get(f'http://api.conceptnet.io/c/en/{word}').json()
    except:
        return []

    if 'edges' in obj:
        return list(set(map(lambda val: val['start']['label'].lower(), obj['edges'])))
    return []

#print(conceptnet('badger'))