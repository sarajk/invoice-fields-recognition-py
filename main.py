import enum
from features_extractor import FeaturesExtractor
import microsoft_computer_vision
import datasaver
import features_normalizer
from som import SOM

print('Initializing and training SOM...')
som = SOM(24, 24, 'input/normalized-with-wrong.csv')
print('SOM trained.')

print('Uploading and analyzing image using MCV...')

analyze_results = None
for i in range(3):

    try:
        analyze_results = microsoft_computer_vision.analyze_image('input/IMG_20210720_141331-copy-blurred.jpg')
        break
    except Exception as e:
        print(e)

if analyze_results is None:
    print('Failed to upload image to MCV!')
    exit(1)
print('Image analyzed using MCV.')

print('Extracting key and value fields while mapping all logical combinations...')
features_extractor = FeaturesExtractor()
features_extractor.load_configuration('input/configuration.json')
key_features = features_extractor.extract_key_features(analyze_results)
value_features = features_extractor.extract_value_features(analyze_results, key_features, True)
print('Features extracted and generated {} combinations'.format(len(value_features)))
normalized_features = features_normalizer.normalize(value_features)

datasaver.write_csv(value_features, 'output/output.csv')
datasaver.write_csv(normalized_features, 'output/output-normalized.csv')

print('Classification started...')
classification_results = som.classify('output/output-normalized.csv')
print('Data classified.')

result = {}

for classification_result in classification_results:

    label = classification_result['label']

    if label not in result:
        result[label] = []
    
    result[label].append(classification_result)

for key in result.keys():

    print(key,':')

    records = result[key]

    for i, record in enumerate(records):

        print('Field {:2d}: {}\t(Confidence: {:.2f}%)'.format(i+1, record['text'], record['confidence']))
        
    print('-----------------------------------------------')