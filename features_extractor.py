import json
from bounding_box import BoundingBox
import math
import re

def clean(text):
    return re.sub(r'[\.,:;\-+=\?]', '', text)

def remove_start(text, remove):

    if text is None or len(text.strip()) == 0:
        return text
    if remove is None or len(remove.strip()) == 0:
        return text
    if text.startswith(':'):
        return text[len(remove):]
    return text


def trim(text):
    return remove_start(text.strip(), ':').strip()

class FeaturesExtractor:

    def __init__(self):
        self.keys = []

    def _is_in_any(self, key_features, value_boundaries, text):
        for key_feature in key_features:
            if key_feature['boundingBox'].boundaries == value_boundaries:
                return True
            if text.strip().lower() == key_feature['key']['label'].strip().lower():
                return True
        return False

    def _to_value_feature(self, key_feature, value_bounding_box, text):

        bounding_box = key_feature['boundingBox']
        
        valueX = value_bounding_box.x
        valueY = value_bounding_box.y
        valueWidth = value_bounding_box.width
        valueHeight = value_bounding_box.height

        keyX = bounding_box.x
        keyY = bounding_box.y
        keyWidth = bounding_box.width
        keyHeight = bounding_box.height

        value_feature = {
            'boundingBox' : value_bounding_box,
            'keyFeature':key_feature,
            'text':text,
            'distance': math.sqrt(math.pow(valueX - keyX, 2) + math.pow(valueY - keyY, 2))
        }

        if ((valueX + valueWidth / 2) < keyX and (valueY + valueHeight) < keyY) :
            value_feature['position'] = 1 # ABOVE_LEFT
        elif (valueX > (keyX + keyWidth / 2) and (valueY + valueHeight) < keyY) :
            value_feature['position'] = 2 # ABOVE_RIGHT
        elif ((valueX + valueWidth / 2) < keyX and (keyY + keyHeight) < valueY) :
            value_feature['position'] = 6 # BELOW_LEFT
        elif (valueX > (keyX + keyWidth / 2) and (keyY + keyHeight) < valueY) :
            value_feature['position'] = 7 # BELOW_RIGHT
        elif ((valueX + valueWidth) < keyX) :
            value_feature['position'] = 3 # LEFT
        elif ((valueY + valueHeight) < keyY) :
            value_feature['position'] = 0 # ABOVE
        elif (valueX > (keyX + keyWidth)) :
            value_feature['position'] = 4 # RIGHT
        else :
            value_feature['position'] = 5 # BELOW
        
        return value_feature

    def load_configuration(self, config_file_path):

        with open(config_file_path, 'r') as json_file:
            self.keys = json.load(json_file)
        
    def extract_key_features(self, analyze_results):

        key_features = []

        if self.keys is None or len(self.keys) == 0:
            raise RuntimeError('Did you forget to call load_configuration? There aren\'t any keys loaded!')
        
        for page_result in analyze_results.read_results:

            for line in page_result.lines:
                boundaries = line.bounding_box
                text = line.text

                for item_header in self.keys:
                    key = item_header['key']

                    if text.strip().lower() == key.strip().lower():
                        key_features.append({
                            'key' : item_header,
                            'boundingBox': BoundingBox(boundaries)
                        })
                    elif  key.strip().lower() in text.strip().lower():
                        for word in line.words:
                            word_text = word.text
                            if word_text.strip().lower() == key.strip().lower():
                                key_features.append({
                                    'key': item_header,
                                    'boundingBox': BoundingBox(word.bounding_box)
                                })
        return key_features

    def extract_value_features(self, analyze_results, key_features, apply_filter):

        value_features = []

        for page_result in analyze_results.read_results:
            
            for key_feature in key_features:

                key = key_feature['key']
                validation_expression = key['validationExpression']

                for line in page_result.lines:

                    value_boundaries = line.bounding_box
                    text = line.text.strip()
                    clean_text = clean(text).strip()

                    if self._is_in_any(key_features, value_boundaries, text):
                        continue

                    if apply_filter:
                        if not re.search(validation_expression, text):
                            continue

                    if len(clean_text.strip()) == 0:
                        continue

                    value_bounding_box = BoundingBox(value_boundaries)
                    value_feature = self._to_value_feature(key_feature, value_bounding_box, trim(text))

                    value_features.append(value_feature)

                    for word in line.words:

                        word_value_boundaries = word.bounding_box
                        word_text = word.text.strip()
                        clean_word_text = clean(word_text).strip()

                        if self._is_in_any(key_features, word_value_boundaries, word_text):
                            continue

                        if apply_filter:
                            if not re.search(validation_expression, word_text):
                                continue
                        
                        if len(clean_word_text) == 0 or clean_word_text == clean_text:
                            continue

                        word_value_bounding_box = BoundingBox(word_value_boundaries)

                        word_value_feature = self._to_value_feature(key_feature, word_value_bounding_box, trim(word_text))
                        value_features.append(word_value_feature)
        return value_features


