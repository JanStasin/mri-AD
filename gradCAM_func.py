import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix
#import random
import numpy as np

# Define a GradCAM helper class
class gradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.model.eval()  # Make sure the model is in evaluation mode
        self.target_layer = target_layer
        
        # Initialize containers for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks to capture the forward activation and backward gradients
        self.forward_handle = self.target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we extract the first element which is the gradients
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int = None) -> torch.Tensor:
   
        # Forward pass
        output = self.model(input_tensor)
        if target_class is None:
            # Pick the top predicted class for the first (and assumed only) sample.
            target_class = output.argmax(dim=1)[0].item()

        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        
        # Backward pass: compute gradients for the target class
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute the weights - average the gradients across spatial dimensions
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # shape: (B, channels, 1, 1)
        
        # Compute the weighted combination of forward activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # Apply ReLU to ensure only positive influences
        cam = F.relu(cam)
        
        # Normalize the CAM so that its values lie between 0 and 1
        b, c, h, w = cam.shape
        cam = cam.view(b, -1)
        cam_min = cam.min(dim=1, keepdim=True)[0]
        cam_max = cam.max(dim=1, keepdim=True)[0]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(b, 1, h, w)
        
        # Optionally, you can interpolate the heatmap to match the input size.
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        return cam, target_class

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def gradCAMS_saver(val_loader, model, encoded_labels, get_all=False):
    # running gradCAM specific functions
    cams = {}
    samples = {}
    model.eval()
    for i, data in enumerate(val_loader):
        inputs, y_val_temp = data
        #print(inputs.shape, y_val_temp.shape)
        for j in range(inputs.shape[0]):
            target_layer = model.conv_block1[-1]
            grad_cam = gradCAM(model, target_layer)
            single_input = inputs[j].unsqueeze(0)
            cam_hm, pred_class = grad_cam.generate_cam(single_input, target_class=None)
            predicted_label = list(encoded_labels.keys())[list(encoded_labels.values()).index(pred_class)]
            #print(f"Predicted class: {predicted_label}")
            if predicted_label not in cams.keys():
                cams[predicted_label] = [cam_hm.numpy()]
            else:
                cams[predicted_label].append(cam_hm.numpy())

            if predicted_label not in samples.keys():
                samples[predicted_label] = inputs[j]

    class_cams = {}
    for key in cams.keys():
        try:
            mean_class_cam = np.mean(cams[key], axis=0)
            #print(mean_class_cam.shape)
            class_cams[key] = mean_class_cam
        except:
            print(cams[key])
            print(len(cams[key]))
            print(f'Error at averaging cams.. class_cams will be empty')

    if get_all:
        return class_cams, cams, samples
    else:
        return class_cams, samples