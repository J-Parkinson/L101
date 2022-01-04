import re
from my_code.load_and_tokenize.loading_helpers import loadGazetteer

dict = loadGazetteer()

# feature 1: is the first character a capital letter?
def title_case(tok):
  return tok[0].isupper()

# feature 2: is it likely an acronym / all caps?
def acronym(tok):
  return tok.isupper()

#feature 3: is it an amount of money / contain currencies
def currency(tok):
  return any(map(lambda x: x.decode("utf-8", 'backslashreplace') in tok.lower(), ['£', '$', '€', '฿', '₱', '₦', '₩', '¥', '₹', '₵', '₡']))

#feature 4: is it a percentage?
def percentage(tok):
  return '%' in tok

#feature 5: is the word in our spam gazetteer?
def checkInDictionary(tok):
  return any(map(lambda x: x.decode("utf-8", 'backslashreplace') in tok.lower(), dict))

# feature 6: punctuation
def justSymbols(tok):
  return bool(re.match("\s*\W\s*", tok))

#Feature 7: length of tokens
def length(tok):
  return len(tok)

