"""
Implements a ReplayStrategy mixin for getting segmenting images via SAM model.

Uses SAM model:https://github.com/facebookresearch/segment-anything 

Usage:

    class MyReplayStrategy(SAMReplayStrategyMixin):
        ...
"""
from pprint import pformat
from mss import mss
import numpy as np
from openadapt import models
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from loguru import logger
from openadapt.events import get_events
from openadapt.utils import display_event, rows2dicts
from openadapt.models import Recording, Screenshot, WindowEvent
from pathlib import Path
import urllib
import numpy as np
import matplotlib.pyplot as plt

from openadapt.strategies.base import BaseReplayStrategy

CHECKPOINT_URL_BASE = "https://dl.fbaipublicfiles.com/segment_anything/"
CHECKPOINT_URL_BY_NAME = {
    "default": f"{CHECKPOINT_URL_BASE}sam_vit_h_4b8939.pth",
    "vit_l": f"{CHECKPOINT_URL_BASE}sam_vit_l_0b3195.pth",
    "vit_b": f"{CHECKPOINT_URL_BASE}sam_vit_b_01ec64.pth",
}
MODEL_NAME = "default"
CHECKPOINT_DIR_PATH = "./checkpoints"
RESIZE_RATIO = 0.1
SHOW_PLOTS = True


class SAMReplayStrategyMixin(BaseReplayStrategy):
    def __init__(
        self,
        recording: Recording,
        model_name=MODEL_NAME,
        checkpoint_dir_path=CHECKPOINT_DIR_PATH,
    ):
        super().__init__(recording)
        self.sam_model = self._initialize_model(model_name, checkpoint_dir_path)
        self.sam_predictor = SamPredictor(self.sam_model)
        self.sam_mask_generator = SamAutomaticMaskGenerator(self.sam_model)

    def _initialize_model(self, model_name, checkpoint_dir_path):
        checkpoint_url = CHECKPOINT_URL_BY_NAME[model_name]
        checkpoint_file_name = checkpoint_url.split("/")[-1]
        checkpoint_file_path = Path(checkpoint_dir_path, checkpoint_file_name)
        if not Path.exists(checkpoint_file_path):
            Path(checkpoint_dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"downloading {checkpoint_url=} to {checkpoint_file_path=}")
            urllib.request.urlretrieve(checkpoint_url, checkpoint_file_path)
        return sam_model_registry[model_name](checkpoint=checkpoint_file_path)

    def get_screenshot_bbox(self, screenshot: Screenshot, show_plots=SHOW_PLOTS) -> str:
        """
        Get the bounding boxes of objects in a screenshot image with RESIZE_RATIO in XYWH format.

        Args:
            screenshot (Screenshot): The screenshot object containing the image.
            show_plots (bool): Flag indicating whether to display the plots or not. Defaults to SHOW_PLOTS.

        Returns:
            str: A string representation of a list containing the bounding boxes of objects.

        """
        image_resized = resize_image(screenshot.image)
        array_resized = np.array(image_resized)
        masks = self.sam_mask_generator.generate(array_resized)
        bbox_list = []
        for mask in masks:
            bbox_list.append(mask["bbox"])
        if SHOW_PLOTS:
            plt.figure(figsize=(20, 20))
            plt.imshow(array_resized)
            show_anns(masks)
            plt.axis("off")
            plt.show()
        return str(bbox_list)

    def get_click_event_bbox(
        self, screenshot: Screenshot, show_plots=SHOW_PLOTS
    ) -> str:
        """
        Get the bounding box of the clicked object in a resized image with RESIZE_RATIO in XYWH format.

        Args:
            screenshot (Screenshot): The screenshot object containing the image.
            show_plots (bool): Flag indicating whether to display the plots or not. Defaults to SHOW_PLOTS.

        Returns:
            str: A string representation of a list containing the bounding box of the clicked object.
            None: If the screenshot does not represent a click event with the mouse pressed.

        """
        for action_event in screenshot.action_event:
            if action_event.name in "click" and action_event.mouse_pressed == True:
                logger.info(f"click_action_event=\n{action_event}")
                image_resized = resize_image(screenshot.image)
                array_resized = np.array(image_resized)

                # Resize mouse coordinates
                resized_mouse_x = int(action_event.mouse_x * RESIZE_RATIO)
                resized_mouse_y = int(action_event.mouse_y * RESIZE_RATIO)
                # Add additional points around the clicked point
                additional_points = [
                    [resized_mouse_x - 1, resized_mouse_y - 1],  # Top-left
                    [resized_mouse_x - 1, resized_mouse_y],  # Left
                    [resized_mouse_x - 1, resized_mouse_y + 1],  # Bottom-left
                    [resized_mouse_x, resized_mouse_y - 1],  # Top
                    [resized_mouse_x, resized_mouse_y],  # Center (clicked point)
                    [resized_mouse_x, resized_mouse_y + 1],  # Bottom
                    [resized_mouse_x + 1, resized_mouse_y - 1],  # Top-right
                    [resized_mouse_x + 1, resized_mouse_y],  # Right
                    [resized_mouse_x + 1, resized_mouse_y + 1],  # Bottom-right
                ]
                input_point = np.array(additional_points)
                self.sam_predictor.set_image(array_resized)
                input_labels = np.ones(
                    input_point.shape[0]
                )  # Set labels for additional points
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_labels,
                    multimask_output=True,
                )
                best_mask_index = np.argmax(scores)
                best_mask = masks[best_mask_index]
                rows, cols = np.where(best_mask)
                # Calculate bounding box coordinates
                x0 = np.min(cols)
                y0 = np.min(rows)
                x1 = np.max(cols)
                y1 = np.max(rows)
                w = x1 - x0
                h = y1 - y0
                input_box = [x0, y0, w, h]
                if SHOW_PLOTS:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(array_resized)
                    show_mask(best_mask, plt.gca())
                    show_box(input_box, plt.gca())
                    # for point in additional_points :
                    #     show_points(np.array([point]),input_labels,plt.gca())
                    show_points(input_point, input_labels, plt.gca())
                    plt.axis("on")
                    plt.show()
                return input_box
        return []


def resize_image(image: Image) -> Image:
    """
    Resize the given image.

    Args:
        image (PIL.Image.Image): The image to be resized.

    Returns:
        PIL.Image.Image: The resized image.

    """
    new_size = [int(dim * RESIZE_RATIO) for dim in image.size]
    image_resized = image.resize(new_size)
    return image_resized


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=120):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2], box[3]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
