import csv

def write_csv(value_features, file_name):

    if len(value_features) == 0:
        raise RuntimeError('Cannot save file ' + file_name + ' because the training_data_list is empty!')

    with open(file_name, 'w', newline='') as csvfile:
        
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')

        writer.writerow(['label', 'x', 'y', 'text', 'id', 'position', 'width', 'distance'])


        for value_feature in value_features:
            key = value_feature['keyFeature']['key']
            bounding_box = value_feature['boundingBox']
            writer.writerow([
                key['label'],
                bounding_box.x,
                bounding_box.y,
                value_feature['text'],
                key['id'],
                value_feature['position'],
                bounding_box.width,
                value_feature['distance']
            ])