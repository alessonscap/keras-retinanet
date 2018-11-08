import keras
from keras.utils import get_file
from keras.layers import Conv2D,MaxPooling2D,Activation
import keras_applications

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image


class CustomBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return custom_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        print('Nothing to download.')
        return None

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['custom_mini','custom_large']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='iara')

    
def custom_retinanet(num_classes, backbone='custom_boneage', inputs=None, modifier=None,noise_aug_std=None,dropout_rate=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('custom_mini', 'custom_large')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'custom_mini':
        model = custom_mini(input_tensor=inputs,noise_aug_std=noise_aug_std)
    elif backbone == 'custom_large':
        model = custom_large(input_tensor=inputs,noise_aug_std=noise_aug_std)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_names = ["custom_out1", "custom_out2", "custom_out3"]
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    
        

    #print("LAYEEEEERSSS {}".format(layer_outputs))
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs,dropout_rate=dropout_rate, **kwargs)



def custom_mini(input_tensor,noise_aug_std=None):
    if noise_aug_std is not None:
        inputs2=keras.layers.GaussianNoise(stddev=noise_aug_std)(input_tensor)
    else:
        inputs2 = input_tensor
    x = Conv2D(8, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs2)
    x = Activation('relu')(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', name='custom_out1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', name='custom_out2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', name='custom_out3')(x)
    
    return keras.models.Model(inputs=input_tensor, outputs=x)

def custom_large(input_tensor,noise_aug_std=None):
    if noise_aug_std is not None:
        inputs2=keras.layers.GaussianNoise(stddev=noise_aug_std)(input_tensor)
    else:
        inputs2 = input_tensor
    x = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs2)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)
    
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='custom_out1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', name='custom_out2')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D()(x)

    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same', name='custom_out3')(x)
    
    return keras.models.Model(inputs=input_tensor, outputs=x)