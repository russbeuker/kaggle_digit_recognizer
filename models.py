from keras.models import Sequential, model_from_json, Model
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Dropout, MaxPooling2D, Flatten, BatchNormalization, Activation, Input, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D

def load_model(session=None):
    # load the saved model and weights
    with open(session.full_path + 'model.json', 'r') as f:
        modelx = model_from_json(f.read())
    modelx.load_weights(session.full_path + 'model.hdf5')
    return modelx

def create_model_1():
    dropout = 0.50
    input = Input(shape=(28, 28, 1))

    x1 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(input)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x1 = Dropout(dropout)(x1)

    x1 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(32, (5, 5), padding='same', kernel_initializer='he_normal')(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(dropout)(x1)
    x1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x1)
    x1 = Dropout(dropout)(x1)

    x1 = Flatten()(x1)
    x1 = Dense(128, activation='relu', kernel_initializer='he_normal')(x1)
    x1 = Dropout(dropout)(x1)

    output = Dense(10, activation='softmax', kernel_initializer='he_normal')(x1)

    model = Model(inputs=[input], outputs=[output])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=False)
    # model.add(BatchNormalization())
    return model