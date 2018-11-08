"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

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
import keras_resnet
from .se_resnet_ import SEResNet50,SEResNet101,SEResNet154

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
from ..layers.gaussian_noise import GaussianNoiseOur

class SEResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone,noise_aug_std=None):
        super(SEResNetBackbone, self).__init__(backbone)
        self.noise_aug_std=noise_aug_std
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        print('Nothing to download.')
        return None

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['seresnet50', 'seresnet101', 'seresnet154']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='iara')


def resnet_retinanet(num_classes, backbone='seresnet50', inputs=None, modifier=None,noise_aug_std=None,dropout_rate=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('seresnet50', 'seresnet101', 'seresnet154')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        inputs = keras.layers.Input(shape=(None, None, 3))
    
    if noise_aug_std is not None:   
        inputs2=GaussianNoiseOur(stddev=noise_aug_std)(inputs)
    else:
        inputs2 = inputs
    

    # create the resnet backbone
    if backbone == 'seresnet50':
        resnet = SEResNet50(input_tensor=inputs2, include_top=False)
        layer_names = ["activation_23", "activation_41", "activation_49"]
    elif backbone == 'seresnet101':
        resnet = SEResNet101(input_tensor=inputs2, include_top=False)
        layer_names = ["activation_29", "activation_98", "activation_106"]
    elif backbone == 'seresnet154':
        resnet = SEResNet154(input_tensor=inputs2, include_top=False)
        layer_names = ["activation_35", "activation_143", "activation_151"]
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    
    layer_outputs = [resnet.get_layer(name).output for name in layer_names]

    return retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs,dropout_rate=dropout_rate, **kwargs)
