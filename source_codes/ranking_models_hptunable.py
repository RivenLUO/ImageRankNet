"""
    This file contains the hyperparameter-tunable models for the project.
"""

import os
import keras_tuner as kt
from tensorflow import keras
import numpy as np
import csv
import json
from keras import layers, optimizers, regularizers, callbacks, models, Model, Sequential, metrics, backend as K, losses
from keras.applications import vgg19, xception, efficientnet, resnet, inception_resnet_v2, VGG19, Xception, \
    EfficientNetB7, ResNet152V2, InceptionResNetV2


def ranking_alone_model_hp(backbone, optimizer,
                           r_dense_units, r_num_dense,
                           learning_rate,
                           r_sub_dense_mode='parallel', r_dense_dropout_rate=None,
                           r_dense_dropout=None,
                           decay=None,
                           r_l1=None, r_l2=None,
                           backbone_weights='imagenet', img_size=224):
    """

    :param backbone:
    :param optimizer:
    :param r_dense_units:
    :param ranking_loss_percentage:
    :param r_num_dense:
    :param learning_rate:
    :param r_sub_dense_mode:
    :param r_dense_dropout_rate:
    :param r_dense_dropout:
    :param decay:
    :param r_regularization:
    :param backbone_weights:
    :param img_size:
    :return:
    """
    # Input images & labels
    img_left = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_right = layers.Input(shape=(img_size, img_size, 3), name="right_image")
    label = layers.Input(shape=2, name="label")

    # Data augmentation
    data_augmentation = Sequential(
        [
            layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", interpolation="bilinear", ),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2)
        ]
    )

    img_left = data_augmentation(img_left)
    img_right = data_augmentation(img_right)

    # Feature extraction
    global feature_extractor

    if backbone == 'vgg19':
        feature_extractor = VGG19(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = vgg19.preprocess_input(img_left)
        img_right = vgg19.preprocess_input(img_right)

    elif backbone == 'xception':
        feature_extractor = Xception(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = xception.preprocess_input(img_left)
        img_right = xception.preprocess_input(img_right)

    elif backbone == 'efficientnetb7':
        feature_extractor = EfficientNetB7(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = efficientnet.preprocess_input(img_left)
        img_right = efficientnet.preprocess_input(img_right)

    elif backbone == 'resnet152v2':
        feature_extractor = EfficientNetB7(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = resnet.preprocess_input(img_left)
        img_right = resnet.preprocess_input(img_right)

    elif backbone == 'inceptionresnetv2':
        feature_extractor = EfficientNetB7(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = inception_resnet_v2.preprocess_input(img_left)
        img_right = inception_resnet_v2.preprocess_input(img_right)

    feature_left = feature_extractor(img_left)
    feature_right = feature_extractor(img_right)

    # Freeze layers
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Subnetwork for Ranking
    ranking_subnetwork = Sequential(name="Ranking_Subnetwork")
    _ranking_dense_units = r_dense_units
    for id_dense in range(r_num_dense):
        if r_sub_dense_mode == 'parallel':
            ranking_subnetwork.add(
                layers.Dense(_ranking_dense_units, activation='relu', name=f"Ranking_Dense_{id_dense + 1}",
                             activity_regularizer=regularizers.l1_l2(l1=r_l1, l2=r_l2),
                             )
            )
            if r_dense_dropout:
                ranking_subnetwork.add(layers.Dropout(r_dense_dropout_rate, name=f"Ranking_DenseDrop_{id_dense + 1}"), )
        elif r_sub_dense_mode == 'decreasing':
            ranking_subnetwork.add(
                layers.Dense(_ranking_dense_units, activation='relu', name=f"Ranking_Dense_{id_dense + 1}",
                             activity_regularizer=regularizers.l1_l2(l1=r_l1, l2=r_l2),
                             )
            )
            if r_dense_dropout:
                ranking_subnetwork.add(layers.Dropout(r_dense_dropout_rate, name=f"Ranking_DenseDrop_{id_dense + 1}"), )
        _ranking_dense_units = _ranking_dense_units // 2
    ranking_subnetwork.add(layers.Dense(1, activation='linear', name="Ranking_Score"))

    feature_left = layers.Flatten()(feature_left)
    feature_right = layers.Flatten()(feature_right)
    x_left_score = ranking_subnetwork(feature_left)
    x_right_score = ranking_subnetwork(feature_right)
    ranking_score_pair = layers.concatenate([x_left_score, x_right_score], name="Ranking_Score_Pair")

    # Construct Model
    ranking_alone_model = Model(inputs=[img_left, img_right], outputs=[ranking_score_pair],
                                name="Ranking_Model")

    # Define ranking loss function
    def ranking_loss(y_true, y_pred):
        print(y_true.shape, y_pred.shape)
        # convert 0s in y_true to -1s
        y_true = K.cast(y_true, dtype='float32')
        y_true = K.cast(K.equal(y_true, 1), dtype='float32') * 2 - 1
        # ranking loss (to satisfy the constraint that the score of the left image should be higher than the right one)
        ranking_loss_value = \
            K.sum(
                K.square(
                    K.maximum(0.0, y_true[:, 0] * (y_pred[:, 0] - y_pred[:, 1]))
                )
            )
        return ranking_loss_value

    # Define metrics
    def ranking_accuracy(y_true, y_pred):
        y_true = K.cast(K.argmax(y_true, axis=1), 'int64')
        pred_diff = y_pred[:, 0] - y_pred[:, 1]
        pred_class = K.cast(pred_diff > 0, 'int64')
        return K.mean(K.equal(y_true, pred_class))

    # Compile model
    if optimizer == 'sgd':
        sgd = optimizers.SGD(learning_rate=learning_rate, decay=decay, momentum=0.2, nesterov=True)
        ranking_alone_model.compile(
            loss={
                'Ranking_Score_Pair': ranking_loss
            },
            optimizer=sgd,
            metrics={
                'Ranking_Score_Pair': [ranking_accuracy]
            }
        )
    elif optimizer == 'adam':
        adam = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        ranking_alone_model.compile(
            loss={
                'Ranking_Score_Pair': ranking_loss
            },
            optimizer=adam,
            metrics={
                'Ranking_Score_Pair': [ranking_accuracy]
            }
        )

    return ranking_alone_model


def comparison_ranking_model_hp(backbone, optimizer,
                                c_num_convs, c_conv_units, c_num_dense, c_dense_units, c_dense,
                                r_dense_units, ranking_loss_percentage, r_num_dense,
                                learning_rate, c_conv_dropout_rate, c_dense_dropout_rate,
                                c_sub_conv_mode='parallel', c_sub_dense_mode='parallel',
                                r_sub_dense_mode='parallel', r_dense_dropout_rate=None,
                                c_conv_dropout=None, r_dense_dropout=None,
                                c_dense_dropout=None,
                                decay=None,
                                r_regularization=None,
                                backbone_weights='imagenet', img_size=224):
    """

    :param optimizer:
    :param weights:
    :param backbone: backbone to use as feature extraction (vgg19, xception, efficientnetb7)
    :param img_size: size of the input images
    :return:
    """
    # Input images & labels
    img_left = layers.Input(shape=(img_size, img_size, 3), name="left_image")
    img_right = layers.Input(shape=(img_size, img_size, 3), name="right_image")
    label = layers.Input(shape=2, name="label")

    # Data augmentation
    data_augmentation = Sequential(
        [
            layers.RandomTranslation(0.2, 0.2, fill_mode="reflect", interpolation="bilinear", ),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.2)
        ]
    )

    img_left = data_augmentation(img_left)
    img_right = data_augmentation(img_right)

    # Feature extraction
    global feature_extractor

    if backbone == 'vgg19':
        feature_extractor = VGG19(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = vgg19.preprocess_input(img_left)
        img_right = vgg19.preprocess_input(img_right)

    elif backbone == 'xception':
        feature_extractor = Xception(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = xception.preprocess_input(img_left)
        img_right = xception.preprocess_input(img_right)

    elif backbone == 'efficientnetb7':
        feature_extractor = EfficientNetB7(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = efficientnet.preprocess_input(img_left)
        img_right = efficientnet.preprocess_input(img_right)

    elif backbone == 'resnet152v2':
        feature_extractor = EfficientNetB7(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = resnet.preprocess_input(img_left)
        img_right = resnet.preprocess_input(img_right)

    elif backbone == 'inceptionresnetv2':
        feature_extractor = EfficientNetB7(weights=backbone_weights, include_top=False, input_shape=(224, 224, 3))
        img_left = inception_resnet_v2.preprocess_input(img_left)
        img_right = inception_resnet_v2.preprocess_input(img_right)

    feature_left = feature_extractor(img_left)
    feature_right = feature_extractor(img_right)

    # Freeze layers
    for layer in feature_extractor.layers[:-4]:
        layer.trainable = False

    # Subnetwork for Comparison
    x = layers.concatenate([feature_left, feature_right])
    _conv_units = c_conv_units
    for id_conv in range(c_num_convs):
        if c_sub_conv_mode == 'parallel':
            x = layers.Conv2D(c_conv_units, (3, 3), padding='valid', name=f"Conv_{id_conv + 1}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu', name=f'Activation_{id_conv + 1}')(x)
            if c_conv_dropout:
                x = layers.Dropout(c_conv_dropout_rate, name=f"ConvDrop_{id_conv + 1}")(x)
        elif c_sub_conv_mode == 'decreasing':
            x = layers.Conv2D(_conv_units, (3, 3), padding='valid', name=f"Conv_{id_conv + 1}")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu', name=f'Activation_{id_conv + 1}')(x)
            if c_conv_dropout:
                x = layers.Dropout(c_conv_dropout_rate, name=f"ConvDrop_{id_conv + 1}")(x)
        _conv_units = _conv_units // 2

    x = layers.Flatten()(x)
    if c_dense == True:
        _dense_units = c_dense_units
        for id_dense in range(c_num_dense):
            if c_sub_dense_mode == 'parallel':
                x = layers.Dense(c_dense_units, activation='relu', name=f"Dense_{id_dense + 1}")(x)
                if c_dense_dropout:
                    x = layers.Dropout(c_dense_dropout_rate, name=f"DenseDrop_{id_dense + 1}")(x)
            elif c_sub_dense_mode == 'decreasing':
                x = layers.Dense(_dense_units, activation='relu', name=f"Dense_{id_dense + 1}")(x)
                if c_dense_dropout:
                    x = layers.Dropout(c_dense_dropout_rate, name=f"DenseDrop_{id_dense + 1}")(x)
            _dense_units = _dense_units // 2

    comparison_output = layers.Dense(2, activation='softmax', name="Comparison_Output")(x)

    # Subnetwork for Ranking
    ranking_subnetwork = Sequential(name="Ranking_Subnetwork")
    _ranking_dense_units = r_dense_units
    for id_dense in range(r_num_dense):
        if r_sub_dense_mode == 'parallel':
            ranking_subnetwork.add(
                layers.Dense(_ranking_dense_units, activation='relu', name=f"Ranking_Dense_{id_dense + 1}",
                             activity_regularizer=r_regularization,
                             kernel_regularizer=r_regularization, )
            )
            if r_dense_dropout:
                ranking_subnetwork.add(layers.Dropout(r_dense_dropout_rate, name=f"Ranking_DenseDrop_{id_dense + 1}"), )
        elif r_sub_dense_mode == 'decreasing':
            ranking_subnetwork.add(
                layers.Dense(_ranking_dense_units, activation='relu', name=f"Ranking_Dense_{id_dense + 1}",
                             activity_regularizer=r_regularization,
                             kernel_regularizer=r_regularization, )
            )
            if r_dense_dropout:
                ranking_subnetwork.add(layers.Dropout(r_dense_dropout_rate, name=f"Ranking_DenseDrop_{id_dense + 1}"), )
        _ranking_dense_units = _ranking_dense_units // 2
    ranking_subnetwork.add(layers.Dense(1, activation='linear', name="Ranking_Score"))

    feature_left = layers.Flatten()(feature_left)
    feature_right = layers.Flatten()(feature_right)
    x_left_score = ranking_subnetwork(feature_left)
    x_right_score = ranking_subnetwork(feature_right)
    ranking_score_pair = layers.concatenate([x_left_score, x_right_score], name="Ranking_Score_Pair")

    # Construct Model
    comparison_model = Model(inputs=[img_left, img_right], outputs=[comparison_output, ranking_score_pair],
                             name="Comparison_Ranking_Model")

    # Define ranking loss function
    def ranking_loss(y_true, y_pred):
        print(y_true.shape, y_pred.shape)
        # convert 0s in y_true to -1s
        y_true = K.cast(y_true, dtype='float32')
        y_true = K.cast(K.equal(y_true, 1), dtype='float32') * 2 - 1
        # ranking loss (to satisfy the constraint that the score of the left image should be higher than the right one)
        ranking_loss_value = \
            K.sum(
                K.square(
                    K.maximum(0.0, y_true[:, 0] * (y_pred[:, 0] - y_pred[:, 1]))
                )
            )
        return ranking_loss_value

    # Define metrics
    def ranking_accuracy(y_true, y_pred):
        y_true = K.argmax(y_true, axis=1)  # 将二元标签转换为类别标签
        pred_diff = y_pred[:, 0] - y_pred[:, 1]  # 计算左图得分减去右图得分
        pred_class = K.cast(pred_diff > 0, 'int64')  # 如果左图得分大于右图得分，则预测为 1，否则预测为 0
        return K.mean(K.equal(y_true, pred_class))  # 计算预测正确的比例

    # Compile model
    if optimizer == 'sgd':
        sgd = optimizers.SGD(learning_rate=learning_rate, decay=decay, momentum=0.2, nesterov=True)
        comparison_model.compile(
            loss={
                'Comparison_Output': 'categorical_crossentropy',
                'Ranking_Score_Pair': ranking_loss
            },
            loss_weights={
                'Comparison_Output': (1.0 - ranking_loss_percentage),
                'Ranking_Score_Pair': ranking_loss_percentage
            },
            optimizer=sgd,
            metrics={
                'Comparison_Output': 'accuracy',
                'Ranking_Score_Pair': [ranking_accuracy]
            }
        )
    elif optimizer == 'adam':
        adam = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        comparison_model.compile(
            loss={
                'Comparison_Output': 'categorical_crossentropy',
                'Ranking_Score_Pair': ranking_loss
            },
            loss_weights={
                'Comparison_Output': (1.0 - ranking_loss_percentage),
                'Ranking_Score_Pair': ranking_loss_percentage
            },
            optimizer=adam,
            metrics={
                'Comparison_Output': 'accuracy',
                'Ranking_Score_Pair': [ranking_accuracy]
            }
        )

    return comparison_model
