import requests

def conceptnet(word):
    obj = requests.get(f'http://api.conceptnet.io/c/en/{word}').json()

    return list(set(map(lambda val: val['start']['label'].lower(), obj['edges'])))

print(conceptnet('badger'))