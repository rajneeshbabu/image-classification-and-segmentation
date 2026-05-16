"""
segmentation/model.py — TEMPLATE
=================================
Implement your semantic segmentation model here.

DO NOT rename this file or the class.
"""

import numpy as np


# Class index mapping (pixel values in segmentation masks)
# 0 = background
# 1 = aeroplane, 2 = bicycle, 3 = bird, 4 = boat, 5 = bottle
# 6 = bus, 7 = car, 8 = cat, 9 = chair, 10 = cow
# 11 = diningtable, 12 = dog, 13 = horse, 14 = motorbike, 15 = person
# 16 = pottedplant, 17 = sheep, 18 = sofa, 19 = train, 20 = tvmonitor
# 255 = ignore / void


class SegmentationModel:
    def __init__(self, weights_dir: str):
        """
        Initialize your model and load trained weights.

        Args:
            weights_dir: absolute path to segmentation/weights/ folder
                         where your saved model files are stored.
        """
        # TODO: Load your trained model here
        # Example:
        #   self.model = torch.load(os.path.join(weights_dir, "best_model.pth"))
        #   self.model.eval()
        raise NotImplementedError("You must implement model loading!")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """
        Predict semantic segmentation mask for the image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), dtype uint8

        Returns:
            Segmentation mask as numpy array, shape (H, W), dtype uint8
            Pixel values should be class indices:
                0 = background
                1-20 = object class (see mapping above)
            
            The output mask MUST be the same spatial size as the input image.
        """
        # TODO: Implement your inference pipeline here
        # 1. Preprocess the image (resize, normalize, etc.)
        # 2. Run through your model
        # 3. Post-process (argmax, resize back to original size, etc.)
        # 4. Return as uint8 numpy array
        raise NotImplementedError("You must implement predict()!")
