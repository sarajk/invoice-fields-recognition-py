MAX_WIDTH = 1165.2672347017815
MIN_WIDTH = 34.22463206816421
MAX_DISTANCE = 3894.4119108196737
MIN_DISTANCE = 277.91305593510504

def normalize(value_features):

    global MAX_WIDTH
    global MIN_WIDTH
    global MAX_DISTANCE
    global MIN_DISTANCE

    for value_feature in value_features:
        bounding_box = value_feature['boundingBox']
        value_feature['distance'] = (value_feature['distance'] - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
        bounding_box.width = (bounding_box.width - MIN_WIDTH) / (MAX_WIDTH - MIN_WIDTH)

    return value_features