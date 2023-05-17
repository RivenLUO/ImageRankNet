from keras import Sequential, layers, Model, optimizers, regularizers, callbacks, metrics
from keras.applications import vgg19, VGG19, Xception, xception, EfficientNetB7, efficientnet, ResNet152V2, resnet, \
    InceptionResNetV2, inception_resnet_v2

data_augmentation = Sequential(
    [
        layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", interpolation="bilinear", ),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2)
    ]
)


def comparison_model_convfusion(backbone, optimizer, learning_rate, dropout=None, decay=None, regularization=None, weights=None, img_size=224):
    """

    :param optimizer:
    :param weights:
    :param backbone: backbone to use as feature extraction (vgg19, xception, efficientnetb7)
    :param img_size: size of the input images
    :return:
    """

    img_left = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_right = layers.Input(shape=(img_size, img_size, 3), name="right_image")

    # Feature extraction
    global feature_extractor

    if backbone == 'vgg19':
        feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = vgg19.preprocess_input(img_left)
        img_right = vgg19.preprocess_input(img_right)

    elif backbone == 'xception':
        feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = xception.preprocess_input(img_left)
        img_right = xception.preprocess_input(img_right)

    elif backbone == 'efficientnetb7':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = efficientnet.preprocess_input(img_left)
        img_right = efficientnet.preprocess_input(img_right)

    elif backbone == 'resnet152v2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = resnet.preprocess_input(img_left)
        img_right = resnet.preprocess_input(img_right)

    elif backbone == 'inceptionresnetv2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = inception_resnet_v2.preprocess_input(img_left)
        img_right = inception_resnet_v2.preprocess_input(img_right)

    img_left = data_augmentation(img_left)
    img_right = data_augmentation(img_right)

    feature_left = feature_extractor(img_left)
    feature_right = feature_extractor(img_right)

    # Freeze layers
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    concat = layers.concatenate([feature_left, feature_right])

    # Subnetwork
    x = layers.Conv2D(512, (3, 3), padding='valid', name="Conv_1")(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_1')(x)
    x = layers.Dropout(0.33, name="Drop_1")(x)

    x = layers.Conv2D(256, (3, 3), padding='valid', name="Conv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_2')(x)
    x = layers.Dropout(0.33, name="Drop_2")(x)

    # x = layers.Conv2D(128, (3, 3), padding='valid', name="Conv_3")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu', name='Activation_3')(x)
    # x = layers.Dropout(0.5, name="Drop_3")(x)

    # x = layers.Conv2D(64, (3, 3), padding='same', name="Conv_4")(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu', name='Activation_4')(x)
    # # x = layers.Dropout(0.5, name="Drop_2")(x)

    x = layers.Flatten()(x)

    output = layers.Dense(2, activation='softmax', name="Final_dense")(x)

    # Model
    comparison_model = Model(inputs=[img_left, img_right], outputs=output)

    if weights:
        comparison_model.load_weights(weights)

    # Compile model
    if optimizer == 'sgd':
        sgd = optimizers.SGD(learning_rate=learning_rate, decay=decay, momentum=0.3, nesterov=True)
        comparison_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    elif optimizer == 'adam':
        adam = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        comparison_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return comparison_model


def q2_comparison_model_convfusion(backbone, weights, optimizer, img_size=224):
    """

    :param optimizer:
    :param weights:
    :param backbone: backbone to use as feature extraction (vgg19, xception, efficientnetb7)
    :param img_size: size of the input images
    :return:
    """

    img_left = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_right = layers.Input(shape=(img_size, img_size, 3), name="right_image")

    # Feature extraction
    global feature_extractor

    if backbone == 'vgg19':
        feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = vgg19.preprocess_input(img_left)
        img_right = vgg19.preprocess_input(img_right)

    elif backbone == 'xception':
        feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = xception.preprocess_input(img_left)
        img_right = xception.preprocess_input(img_right)

    elif backbone == 'efficientnetb7':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = efficientnet.preprocess_input(img_left)
        img_right = efficientnet.preprocess_input(img_right)

    elif backbone == 'resnet152v2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = resnet.preprocess_input(img_left)
        img_right = resnet.preprocess_input(img_right)

    elif backbone == 'inceptionresnetv2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = inception_resnet_v2.preprocess_input(img_left)
        img_right = inception_resnet_v2.preprocess_input(img_right)

    img_left = data_augmentation(img_left)
    img_right = data_augmentation(img_right)

    feature_left = feature_extractor(img_left)
    feature_right = feature_extractor(img_right)

    # Freeze layers
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    concat = layers.concatenate([feature_left, feature_right])

    # Subnetwork
    x = layers.Conv2D(1024, (3, 3), padding='same', name="Conv_1")(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_1')(x)
    # x = layers.Dropout(0.2, name="Drop_1")(x)

    x = layers.Conv2D(512, (3, 3), padding='same', name="Conv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_2')(x)
    # x = layers.Dropout(0.2, name="Drop_2")(x)

    x = layers.Conv2D(256, (3, 3), padding='same', name="Conv_3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_3')(x)
    # x = layers.Dropout(0.2, name="Drop_3")(x)

    x = layers.Conv2D(128, (3, 3), padding='same', name="Conv_4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_4')(x)
    # # x = layers.Dropout(0.2, name="Drop_4")(x)

    x = layers.Flatten()(x)

    # x = layers.Dense(1024, activation='relu', name="Dense_1")(x)
    # # x = layers.Dropout(0.5, name="Drop_1")(x)
    #
    # x = layers.Dense(512, activation='relu', name="Dense_2")(x)
    # # x = layers.Dropout(0.5, name="Drop_2")(x)
    #
    # x = layers.Dense(256, activation='relu', name="Dense_3")(x)
    # # x = layers.Dropout(0.5, name="Drop_3")(x)

    output = layers.Dense(2, activation='softmax', name="Final_dense")(x)

    # Model
    comparison_model = Model(inputs=[img_left, img_right], outputs=output)

    if weights:
        comparison_model.load_weights(weights)

    # Compile model
    if optimizer == 'sgd':
        sgd = optimizers.SGD(learning_rate=1e-5, decay=1e-6, momentum=0.3, nesterov=True)
        comparison_model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy',]
        )
    elif optimizer == 'adam':
        adam = optimizers.Adam(learning_rate=5e-6, decay=1e-6)
        comparison_model.compile(
            loss='categorical_crossentropy',
            optimizer=adam,
            metrics=['accuracy']
        )


    return comparison_model


def q3_comparison_model_convfusion(backbone, weights, img_size=224):
    """

    :param weights:
    :param backbone: backbone to use as feature extraction (vgg19, xception, efficientnetb7)
    :param img_size: size of the input images
    :return:
    """

    img_left = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_right = layers.Input(shape=(img_size, img_size, 3), name="right_image")

    # Feature extraction
    global feature_extractor

    if backbone == 'vgg19':
        feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = vgg19.preprocess_input(img_left)
        img_right = vgg19.preprocess_input(img_right)

    elif backbone == 'xception':
        feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = xception.preprocess_input(img_left)
        img_right = xception.preprocess_input(img_right)

    elif backbone == 'efficientnetb7':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = efficientnet.preprocess_input(img_left)
        img_right = efficientnet.preprocess_input(img_right)

    elif backbone == 'resnet152v2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = resnet.preprocess_input(img_left)
        img_right = resnet.preprocess_input(img_right)

    elif backbone == 'inceptionresnetv2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = inception_resnet_v2.preprocess_input(img_left)
        img_right = inception_resnet_v2.preprocess_input(img_right)

    img_left = data_augmentation(img_left)
    img_right = data_augmentation(img_right)

    feature_left = feature_extractor(img_left)
    feature_right = feature_extractor(img_right)

    # Freeze layers
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    concat = layers.concatenate([feature_left, feature_right])

    # Subnetwork
    x = layers.Conv2D(1024, (3, 3), padding='same', name="Conv_1")(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_1')(x)
    # x = layers.Dropout(0.66, name="Drop_1")(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name="Conv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_2')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name="Conv_3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_3')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name="Conv_4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_4')(x)
    # x = layers.Dropout(0.5, name="Drop_2")(x)
    x = layers.Flatten()(x)
    output = layers.Dense(2, activation='softmax', name="Final_dense")(x)

    # Model
    comparison_model = Model(inputs=[img_left, img_right], outputs=output)

    if weights:
        comparison_model.load_weights(weights)

    # Compile model
    sgd = optimizers.SGD(learning_rate=1e-5, decay=1e-6, momentum=0.3, nesterov=True)
    comparison_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return comparison_model


def q4_comparison_model_convfusion(backbone, weights, img_size=224):
    """

    :param weights:
    :param backbone: backbone to use as feature extraction (vgg19, xception, efficientnetb7)
    :param img_size: size of the input images
    :return:
    """

    img_left = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_right = layers.Input(shape=(img_size, img_size, 3), name="right_image")

    # Feature extraction
    global feature_extractor

    if backbone == 'vgg19':
        feature_extractor = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = vgg19.preprocess_input(img_left)
        img_right = vgg19.preprocess_input(img_right)

    elif backbone == 'xception':
        feature_extractor = Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = xception.preprocess_input(img_left)
        img_right = xception.preprocess_input(img_right)

    elif backbone == 'efficientnetb7':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = efficientnet.preprocess_input(img_left)
        img_right = efficientnet.preprocess_input(img_right)

    elif backbone == 'resnet152v2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = resnet.preprocess_input(img_left)
        img_right = resnet.preprocess_input(img_right)

    elif backbone == 'inceptionresnetv2':
        feature_extractor = EfficientNetB7(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        img_left = inception_resnet_v2.preprocess_input(img_left)
        img_right = inception_resnet_v2.preprocess_input(img_right)

    img_left = data_augmentation(img_left)
    img_right = data_augmentation(img_right)

    feature_left = feature_extractor(img_left)
    feature_right = feature_extractor(img_right)

    # Freeze layers
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    concat = layers.concatenate([feature_left, feature_right])

    # Subnetwork
    x = layers.Conv2D(1024, (3, 3), padding='same', name="Conv_1")(concat)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_1')(x)
    # x = layers.Dropout(0.66, name="Drop_1")(x)
    x = layers.Conv2D(512, (3, 3), padding='same', name="Conv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_2')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name="Conv_3")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_3')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name="Conv_4")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu', name='Activation_4')(x)
    # x = layers.Dropout(0.5, name="Drop_2")(x)
    x = layers.Flatten()(x)
    output = layers.Dense(2, activation='softmax', name="Final_dense")(x)

    # Model
    comparison_model = Model(inputs=[img_left, img_right], outputs=output)

    if weights:
        comparison_model.load_weights(weights)

    # Compile model
    sgd = optimizers.SGD(learning_rate=1e-5, decay=1e-6, momentum=0.3, nesterov=True)
    comparison_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return comparison_model
