import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import csv
import pickle
import chardet
import sys
import json

defined_colors = ['red', 'green', 'yellow', 'blue', '#7fffd4', '#4b5320',
                  '#ff2052', '#480607', '#cc5500', '#00bfff', '#ff3800', '#00008b', '#ccff00', '#dcdcdc',
                  '#a8e4a0', '#fba0e3']

def get_data(filename):

    content = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',', quotechar='"')
        for row in reader:
            content.append(row)

    columnsDict = {}
    for i,col in enumerate(content[0]):
        columnsDict[i] = col

    training_data = []
    for columns in content[1:]:

        row_dict = {}
        for i,col in enumerate(columns):
            row_dict[columnsDict[i]] = col

        if(row_dict['label'] == 'Incorrect'):
            row_dict['color'] = 'black'
        else:
            row_dict['color'] = defined_colors[int(row_dict['id'])]

        training_data.append(row_dict)
    return training_data


class SOM:
    def __init__(self, n_rows, n_columns, training_file_path) -> None:
        self.data = []
        self.colors = defined_colors
        self.labels = []
        
        training_data = get_data(training_file_path)
        # print(training_data)


        for row_dict in training_data:
            record = [int(row_dict['id']),float(row_dict['distance']), int(row_dict['position']), float(row_dict['width'])]
            self.colors.append(row_dict['color'])
            self.labels.append(row_dict['label'])
            self.data.append(record)

        self.data = np.float32(self.data)

        self.som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)

        # som.load_umatrix('umatrix.txt')
        # som.load_bmus('bmus.npy')
        # som.load_codebook('codebook.npy')
        self.som.train(self.data)
        self.som.cluster()

        # NOTE: Uncomment this line to view the umatrix
        # self.som.view_umatrix(bestmatches=True, bestmatchcolors=self.colors, labels=self.labels)


    def _get_bmu(self, test_data):

        full_arr = []

        for row_dict in test_data:
            full_arr.append([int(row_dict['id']),float(row_dict['distance']), int(row_dict['position']), float(row_dict['width'])])

        arr = np.float32(full_arr)

        activation_map = self.som.get_surface_state(data=arr)
        bmus = self.som.get_bmus(activation_map)

        result = []

        for bmu_match in bmus:

            result_dict = {}

            for i,bmu in enumerate(list(self.som.bmus)):
                if(list(bmu)==(list(bmu_match))):
                    if self.labels[i].lower() not in result_dict:
                        result_dict[self.labels[i].lower()] = 0
                    result_dict[self.labels[i].lower()]+=1
            result.append(result_dict)

        # print('Data might be any of these:' , result, 'for', row_dict['text'])

        confidence_list = []

        for result_dict in result:
            if len(result_dict) == 0:
                confidence_list.append( None)
            else:

                best_confidence = {
                    'key'  :None,
                    'confidence' :0.0
                }


                total = 0
                for value in result_dict.values():
                    total+=value

                for key in result_dict.keys():

                    confidence = (result_dict[key]/total)*100

                    if best_confidence['confidence'] < confidence:
                        best_confidence['key'] = key
                        best_confidence['confidence'] = confidence
                if best_confidence['key'] == 'incorrect':
                    confidence_list.append(None)
                else:
                    confidence_list.append(best_confidence)
        return confidence_list


    def classify(self, test_data_file_path):
        test_data = get_data(test_data_file_path)

        result_list = self._get_bmu(test_data)

        classified_data = []
        for i,result_dict in enumerate(result_list):

            row_dict = test_data[i]

            if result_dict is None:
                continue

            classified_data.append({
                'label': result_dict['key'],
                'text': row_dict['text'],
                'confidence': result_dict['confidence'],
                'x': row_dict['x'],
                'y': row_dict['y']
            })
        return classified_data