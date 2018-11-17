import keras
from sklearn.metrics import classification_report
from keras.utils import to_categorical


def check_preds(x, y):
    trained_model = keras.models.load_model('trained_model.h5')
    predictions = trained_model.predict_classes(x)

    print(classification_report(y, to_categorical(predictions)))
