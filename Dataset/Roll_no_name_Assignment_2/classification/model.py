"""
classification/model.py — TEMPLATE
====================================
Implement your multilabel classification model here.

DO NOT rename this file or the class.
"""

import numpy as np

class ClassificationModel:
    def __init__(self, weights_dir: str):
        """
        Initialize your model and load trained weights.

        Args:
            weights_dir: absolute path to classification/weights/ folder
                         where your saved model files are stored.
        """
        # TODO: Load your trained model here
        # Example:
        #   self.model = torch.load(os.path.join(weights_dir, "best_model.pth"))
        #   self.model.eval()
        raise NotImplementedError("You must implement model loading!")

    def predict(self, image: np.ndarray) -> dict:
        """
        Predict which of the 20 classes are present in the image.

        Args:
            image: RGB image as numpy array, shape (H, W, 3), dtype uint8

        Returns:
            dict mapping class_name (str) -> probability (float in [0, 1])
            Must contain ALL 20 classes listed in dataset_info.json.

        Example output:
            {
                "aeroplane": 0.02, "bicycle": 0.95, "bird": 0.01,
                "boat": 0.03, "bottle": 0.10, "bus": 0.00,
                "car": 0.12, "cat": 0.01, "chair": 0.04,
                "cow": 0.00, "diningtable": 0.01, "dog": 0.02,
                "horse": 0.00, "motorbike": 0.03, "person": 0.88,
                "pottedplant": 0.01, "sheep": 0.00, "sofa": 0.02,
                "train": 0.01, "tvmonitor": 0.05
            }
        """
        # TODO: Implement your inference pipeline here
        # 1. Preprocess the image (resize, normalize, etc.)
        # 2. Run through your model
        # 3. Return predictions as a dict
        raise NotImplementedError("You must implement predict()!")
