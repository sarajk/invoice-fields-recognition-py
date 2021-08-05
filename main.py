import enum
from features_extractor import FeaturesExtractor
import microsoft_computer_vision
import datasaver
import features_normalizer
from som import SOM
import os
import image_drawing

features_configuration_file_path = 'input/configuration.json'
training_data_file = 'input/normalized-with-wrong.csv'

# NOTE: YOU NEED TO SPECIFY HERE THE IMAGE FILE PATH, OR KEEP THE PROVIDED ONE FOR PoC
image_file_path = 'input/IMG_20210720_141331-copy-blurred.jpg'

#region: Files Definition
image_file_name, image_file_extension = os.path.splitext(os.path.basename(image_file_path))
image_features_output_dir = 'output/analysis/' + image_file_name
image_features_output_file = image_features_output_dir + '/output.csv'
image_features_output_normalized_file = image_features_output_dir + '/output-normalized.csv'
image_output_file_path = image_features_output_dir + '/'+ os.path.basename(image_file_path)

if not os.path.exists(image_features_output_dir):
    # Create a directory with the image name
    os.makedirs(image_features_output_dir)

#endregion

#region: Training
print('Initializing and training SOM...')
som = SOM(24, 24, training_data_file)
print('SOM trained.')
#endregion

#region: MCV OCR
print('Uploading and analyzing image using MCV...')

analyze_results = None
for i in range(3):

    try:
        analyze_results = microsoft_computer_vision.analyze_image(image_file_path)
        break
    except Exception as e:
        print(e)

if analyze_results is None:
    print('Failed to upload image to MCV!')
    exit(1)
print('Image analyzed using MCV.')
#endregion

#region: Drawing
print('Drawing bounding boxes on image based on the OCR findings...')
image_drawing.draw(analyze_results, image_file_path, image_output_file_path)
print('Bounding boxes drawn and saved onto', image_output_file_path)
#endregion

#region: Features Extraction
print('Extracting key and value fields while mapping all logical combinations...')
features_extractor = FeaturesExtractor()
features_extractor.load_configuration(features_configuration_file_path)
key_features = features_extractor.extract_key_features(analyze_results)
value_features = features_extractor.extract_value_features(analyze_results, key_features, True)
print('Features extracted and generated {} combinations'.format(len(value_features)))
normalized_features = features_normalizer.normalize(value_features)

datasaver.write_csv(value_features, image_features_output_file)
datasaver.write_csv(normalized_features, image_features_output_normalized_file)
#endregion

#region: Classification
print('Classification started...')
classification_results = som.classify(image_features_output_normalized_file)
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
#endregion