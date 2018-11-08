"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


import keras
from keras.utils import get_file
import keras_applications

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
from ..layers.gaussian_noise import GaussianNoiseOur


class InceptionBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return inception_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        if self.backbone == 'inception_resnet_v2':
            resource = keras_applications.inception_resnet_v2.BASE_WEIGHT_URL+'inception_resnet_v2_weights_'+'tf_dim_ordering_tf_kernels_notop.h5'
            checksum = 'd19885ff4a710c122648d3b5c3b684e4'
        elif self.backbone == 'inception_v3':
            resource = keras_applications.inception_v3.WEIGHTS_PATH_NO_TOP
            checksum = 'bcbd6486424b2319ff4ef7d526e38f63'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['inception_resnet_v2', 'inception_v3']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='tf')

def inception_retinanet(num_classes, backbone='inception_v3', inputs=None, modifier=None,noise_aug_std=None,dropout_rate=None, **kwargs):
    """ Constructs a retinanet model using a inception backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('inception_resnet_v2', 'inception_v3')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a inception backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
    if noise_aug_std is not None:   
        inputs2=GaussianNoiseOur(stddev=noise_aug_std)(inputs)
    else:
        inputs2 = inputs
        
    # create the inception backbone
    if backbone == 'inception_resnet_v2':
        #raise ValueError("Backbone '{}' not working right now.".format(backbone))
        inc = keras.applications.InceptionResNetV2(input_tensor=inputs2, include_top=False)
        layer_names = ["block8_8_ac", "block8_9_ac", "conv_7b_ac"] ##for v2
    elif backbone == 'inception_v3':
        #raise ValueError("Backbone '{}' not working right now.".format(backbone))
        inc = keras.applications.InceptionV3(input_tensor=inputs2, include_top=False)
        layer_names = ["mixed8","mixed9","mixed10"] #for v3
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        inc = modifier(inc)

    # create the full model
    
    layer_outputs = [inc.get_layer(name).output for name in layer_names]
    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs,dropout_rate=dropout_rate, **kwargs)