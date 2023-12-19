'''
    Boeing Proprietary.
    Developed by Nobal Niraula, Boeing Research & Technology
'''

from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import re
import string
import os
import datetime as dt
import time
import logging
from logging.handlers import TimedRotatingFileHandler

DEPRESSURIZATION = 'depressurization'
DEGRADED_CONTROLLABILITY= 'degraded-controllability'
CORROSION_LIMIT = 'corrosion-limit'

LOG_FILE = os.getcwd() + "/logs"
if not os.path.exists(LOG_FILE):
    os.makedirs(LOG_FILE)
LOG_FILE = LOG_FILE + "/" + dt.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d') + ".log"
logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
fileHandler = TimedRotatingFileHandler("{0}".format(LOG_FILE), when='D', interval=1)
# fileHandler = logging.FileHandler("{0}".format(LOG_FILE))
fileHandler.setFormatter(logFormatter)
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)
rootLogger.setLevel(logging.INFO)

class PreprocessingUtils:


    def __init__(self):
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.punctuation_remove_regex = re.compile('[' + string.punctuation + ']')


    def remove_stop_words(self, inp, protected_terms=[]):
        tokens = None
        if isinstance(inp, str):
            tokens = inp.split()
            return " ".join([x for x in tokens if x in protected_terms or x not in self.stop_words])
        elif isinstance(inp, list):
            return [x for x in tokens if x in protected_terms or x not in self.stop_words]
        if not tokens:
            raise ValueError("Invalid input. Requires string or a list")

        return tokens

    @staticmethod
    def remove_these_words(inp, words_to_be_removed):
        tokens = None
        if isinstance(inp, str):
            tokens = inp.split()
            return " ".join([x for x in tokens if x not in words_to_be_removed])
        elif isinstance(inp, list):
            return [x for x in tokens if x not in words_to_be_removed]
        if not tokens:
            raise ValueError("Invalid input. Requires string or a list")

        return tokens

    def preprocess_records(self, prep_config, records):
        '''
        :param prep_config:
        :param records: list of text
        :return:
        '''

        # Lower case
        if "lower_case" in prep_config and prep_config["lower_case"]:
            records = [x.lower() for x in records]

        # Remove punctuations
        if "remove_punctuations" in prep_config and prep_config["remove_punctuations"]:
            records = [re.sub(self.punctuation_remove_regex, ' ', x) for x in records]

        # Remove white spaces
        records = [re.sub("\\s+", " ", x) for x in records]

        if "use_mad_preprocessing" in prep_config and prep_config["use_mad_preprocessing"]:
            raise Exception("Not implemented yet")

        if "remove_duplicate_sentences" in prep_config and prep_config["remove_duplicate_sentences"]:
            raise Exception("Not implemented yet")

        if "remove_general_stop_words" in prep_config and prep_config["remove_general_stop_words"]:
            records = [self.remove_stop_words(x, protected_terms=[]) for x in records]

        if "custom_stopwords" in prep_config and prep_config["custom_stopwords"]:
            records = [PreprocessingUtils.remove_these_words(x, prep_config["custom_stopwords"]) for x in records]

        # Remove singleton punctuations
        records = [PreprocessingUtils.remove_these_words(x, string.punctuation) for x in records]

        return records