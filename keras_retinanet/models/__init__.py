from ..layers.gaussian_noise import GaussianNoiseOur

class Backbone(object):
    """ This class stores additional information on backbones.
    """
    def __init__(self, backbone, noise_aug_std=None):
        # a dictionary mapping custom layer names to the correct classes
        from .. import layers
        from .. import losses
        from .. import initializers
        self.custom_objects = {
            'UpsampleLike'     : layers.UpsampleLike,
            'PriorProbability' : initializers.PriorProbability,
            'RegressBoxes'     : layers.RegressBoxes,
            'FilterDetections' : layers.FilterDetections,
            'Anchors'          : layers.Anchors,
            'ClipBoxes'        : layers.ClipBoxes,
            '_smooth_l1'       : losses.smooth_l1(),
            '_focal'           : losses.focal(),
        }

        self.backbone = backbone
        self.noise_aug_std = noise_aug_std
        self.validate()

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        raise NotImplementedError('retinanet method not implemented.')

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        """
        raise NotImplementedError('download_imagenet method not implemented.')

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        raise NotImplementedError('validate method not implemented.')

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        Having this function in Backbone allows other backbones to define a specific preprocessing step.
        """
        raise NotImplementedError('preprocess_image method not implemented.')


def backbone(backbone_name,noise_aug_std=None):
    """ Returns a backbone object for the given backbone.
    """
    if 'inception' in backbone_name:
        from .inception import InceptionBackbone as b
    elif 'seresnet' in backbone_name:
        from .seresnet import SEResNetBackbone as b
    elif 'resnet' in backbone_name:
        from .resnet import ResNetBackbone as b
    elif 'mobilenet' in backbone_name:
        from .mobilenet import MobileNetBackbone as b
    elif 'vgg' in backbone_name:
        from .vgg import VGGBackbone as b
    elif 'densenet' in backbone_name:
        from .densenet import DenseNetBackbone as b
    elif 'custom' in backbone_name:
        from .custom_model import CustomBackbone as b
    else:
        raise NotImplementedError('Backbone class for  \'{}\' not implemented.'.format(backbone))
    print('Chosen backbone {}'.format(backbone_name))
    return b(backbone_name,noise_aug_std)


def load_model(filepath, backbone_name='resnet50', convert=False, nms=True, class_specific_filter=True, nms_threshold=0.5, anchors_ratios=None, anchors_scales=None):
    """ Loads a retinanet model using the correct custom objects.

    # Arguments
        filepath: one of the following:
            - string, path to the saved model, or
            - h5py.File object from which to load the model
        backbone_name         : Backbone with which the model was trained.
        convert               : Boolean, whether to convert the model to an inference model.
        nms                   : Boolean, whether to add NMS filtering to the converted model. Only valid if convert=True.
        class_specific_filter : Whether to use class specific filtering or filter for the best scoring class only.
        anchors_ratios        : The array of anchors ratios.
        anchors_scales        : The array of anchors scales.

    # Returns
        A keras.models.Model object.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    import keras.models
    iara_custom_objs = backbone(backbone_name).custom_objects
    iara_custom_objs['relu6']=keras.layers.ReLU(6.0)
    model = keras.models.load_model(filepath, custom_objects=iara_custom_objs)
    if convert:
        from .retinanet import retinanet_bbox
        model = retinanet_bbox(model=model, nms=nms, class_specific_filter=class_specific_filter, nms_threshold = nms_threshold,anchors_ratios=anchors_ratios,anchors_scales=anchors_scales)

    return model
