import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize.regexp import RegexpTokenizer
from nltk.corpus import stopwords
import langid
from bs4 import BeautifulSoup
import html
from imblearn.over_sampling import RandomOverSampler, SMOTEN
from libretranslatepy import LibreTranslateAPI
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier



##import original data
##data is already split into Training/Testing, no need to re-split
Rak_train_raw = pd.read_csv('../data/raw/X_train_update.csv', index_col=0)  ##raw X train data
RakY_train_raw = pd.read_csv('../data/raw/Y_train_CVw08PX.csv', index_col=0)  ##raw Y train data
Rak_test_raw = pd.read_csv('../data/raw/X_test_update.csv', index_col=0)  ##raw X test data

#####
##Clean-up strings
def rak_data_cleanup(rak_data_raw):
    ##Fix problematic strings for text clean-up
    ##FIXME not sure if deep copy inside function is needed
    rak_data = rak_data_raw.copy(deep = True)

    ## Replace NaN with ''
    ## (for some reason strings can include numeric NaN values)
    rak_data['description'] = rak_data['description'].fillna('')

    ## lower case
    rak_data['designation'] = rak_data['designation'].str.lower()
    rak_data['description'] = rak_data['description'].str.lower()

    ##FIXME simplify syntax
    ##Unescape HTML
    rak_data['designation'] = rak_data.apply(lambda row: html.unescape(row['designation']), axis = 1)
    rak_data['description'] = rak_data.apply(lambda row: html.unescape(row['description']), axis = 1)

    ## Remove HTML tags
    rak_data['designation'] = rak_data.apply(lambda row: BeautifulSoup(row['designation'], "html.parser").get_text(separator=" "), axis = 1)
    rak_data['description'] = rak_data.apply(lambda row: BeautifulSoup(row['description'], "html.parser").get_text(separator=" "), axis = 1)

    ##FIXME drop empty designation & description rows
    ##FIXME make fun to clean-up data (to apply to test as well)


    ##FIXME list of problematic strings to fix
    ## àªtre

    ##Regex replacements
    repl_dict = {
                ##FIXME nltk.classify.textcat.TextCat().remove_punctuation() 
                r'n°': r' numéro ', 

                ##FIXME not sure how to handle '¿' or '?'
                ##insert space around any non-digit, non-word and non-whitespace with (e.g. '\?' -> ' \? ', 'n°' -> 'n ° ')
                ##except ¿'
                r"[^\d\w\s¿\?'\-]": r' \g<0> ',  

                ##FIXME possibly remove digits after translation
                r"\b\S*[0-9]+\S*\b": '' ##drop all words that contain digits (so drop all digits as well)
                }

    rak_data = rak_data.replace(to_replace = {'designation': repl_dict,
                                                'description': repl_dict},
                                    regex = True)



    ##Concat strings
    rak_data['product_txt'] = rak_data['designation'] + ' . -//- ' + rak_data['description']

    rak_data['product_txt_len'] = rak_data['product_txt'].apply(len)

    return rak_data

##clean-up X train data
Rak_train = rak_data_cleanup(Rak_train_raw)

##clean-up X test data
Rak_test = rak_data_cleanup(Rak_test_raw)


#####
##foreign language handling
##detect phrase language using 'langid' on product_txt
def lang_detect(data):
    # langid.set_languages(langs=None)
    langid.set_languages(langs=['fr', 'en', 'de', 'it', 'es', 'pt'])
    data['lang'] = data['product_txt'].apply(lambda x: langid.classify(x)[0])

    return data

##detect languages on Rakuten processed data
Rak_train = lang_detect(data = Rak_train)
Rak_test = lang_detect(data = Rak_test)


##translate using libretranslate (self-hosted process)
##func to obtain translations 
## if rerun == True then run translation code, if False (default) then just load csv from supplied path
def translate_txt(data = None, rerun = False, csv_file = '../data/processed/Rak_train_translations.csv'):
    ##FIXME improve func not to require data if just loading csv
    if rerun == True:
        ##NOTE: need to start external process first (and this code is time-consuming)
        # libretranslate --update-models --load-only fr,en,es,de,it,pt
        # libretranslate --load-only fr,en,es,de,it,pt
        lt = LibreTranslateAPI("http://localhost:5000/")

        ##re-use detected language
        def lt_fun(row):
            if row.name % 5000 == 0: print(row.name)
            # print(row['lang'])
            transl = row['product_txt'] if row['lang'] == 'fr' else lt.translate(
                row['product_txt'], source=row['lang'], target="fr")
            return transl


        data['product_txt_transl'] = data.apply(lambda row: lt_fun(row), axis=1)

        ##save translations to csv (overwrites existing file)
        data.to_csv(csv_file)
    else:
        ##load translations from stored csv file
        data = pd.read_csv(csv_file)

    return data

##translate X train (by default load existing csv file with translations)
Rak_train = translate_txt(data = None, rerun = False, csv_file = '../data/processed/Rak_train_translations.csv')
##FIXME run translations for X test as well


#####
##Training Sample Rebalancing
RakX_train = Rak_train[['product_txt_transl']]

##Oversampling (only on training data)
##SMOTEN
smo = SMOTEN(random_state=27732)
RakX_train_sm, Raky_train_sm = smo.fit_resample(RakX_train, RakY_train_raw)


#####
##Tokenization
##FIXME def tokenization fun to reuse with test data as well

##Stop Words
fr_stop_words = stopwords.words('french')

# Créer un vectorisateur 
##FIXME consider custom tokenizer and max_features
# regexp_tokenizer = RegexpTokenizer("[a-zA-ZÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸàâæçéèêëîïôœùûüÿ]{3,}") ##words with at least 3 characaters
vect_tfidf = TfidfVectorizer(    
    max_features=10000,
    stop_words=fr_stop_words
    # , tokenizer = regexp_tokenizer
)

# Mettre à jour la valeur de X_train_tfidf et X_test_tfidf
##FIXME need to finalize tokenization
RakX_train_sm_tfidf = vect_tfidf.fit_transform(RakX_train_sm['product_txt_transl'])





