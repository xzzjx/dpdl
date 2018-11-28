# coding: utf-8

from keras.models import Sequential
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras import backend as K
K.set_image_dim_ordering('th')

def buildvgg16():
    # 构造VGG16模型
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=(3, 150, 150)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',by_name=True)
    model.summary()

# 在初始化好的VGG网络上添加预训练好的模型
# top_model = Sequential()
# top_model.add(Flatten(input_shape=model.output_shape[1:])) #  (4,4,512)
# top_model.add(Dense(256, activation='relu'))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1, activation='sigmoid'))

# top_model.load_weights('bottleneck_fc_model.h5',by_name=True)
# model.add(top_model)

# 将最后一个卷积块前的卷基层参数冻结,把随后卷积块前的权重设置为不可训练（权重不会更新）
    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                metrics=['accuracy'])
    
    return model
# 以低学习率进行训练
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                    shear_range=0.2,
#                                    zoom_range=0.2,
#                                    horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)
# train_generator = train_datagen.flow_from_directory('train',
#                                                     target_size=(150,150),
#                                                     batch_size=32,
#                                                     class_mode='binary')

# validation_generator = test_datagen.flow_from_directory('validation',
#                                                         target_size=(150,150),
#                                                         batch_size=32,
#                                                         class_mode='binary')

# model.fit_generator(train_generator,
#                     steps_per_epoch=10,
#                     epochs=50,
#                     validation_data=validation_generator,
#                     validation_steps=10)