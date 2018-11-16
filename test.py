import keras
from sklearn.metrics import classification_report
from utils import wav2mfcc, model, get_data
from keras.utils import to_categorical

def check_preds(X, y):

    trained_model = keras.models.load_model('trained_model.h5')
    predictions = trained_model.predict_classes(X)

    print(classification_report(y, to_categorical(predictions)))