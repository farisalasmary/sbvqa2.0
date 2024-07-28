from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import ttach as tta

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection

import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


class MyGradCAM(GradCAM):
    
    def __init__(self, model, target_layers,
                 reshape_transform=None):
        super(
            MyGradCAM,
            self).__init__(
            model,
            target_layers,
            reshape_transform)
    
    def forward(
        self, my_input_tensor: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        image_tensor, (audio_tensor, audio_len) = my_input_tensor
        input_tensor = image_tensor
        image_tensor = image_tensor.to(self.device)
        audio_tensor = audio_tensor.to(self.device)
        audio_len = audio_len.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(my_input_tensor)

        if targets is None:
            target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            loss.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        res = self.aggregate_multi_layers(cam_per_layer)
        self.activations_and_grads.release()
        del outputs
        return res
