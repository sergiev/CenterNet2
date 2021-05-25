# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import pycocotools.mask as mask_util

from detectron2.structures import Boxes, RotatedBoxes
from detectron2.utils.visualizer import (
    ColorMode,
    Visualizer,
    _create_text_labels,
    _PanopticPrediction,
)

from .colormap import random_color


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


def xyxy_abs_to_poly(bbox):
    """
    Args:
        bbox: quadruplet of ints [xmin, ymin, xmax, ymax]

    Returns:
        Polygon in IC15 format (clockwise XY points, starting from left top)
    """
    xmin, ymin, xmax, ymax = bbox
    return xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax


class VideoVisualizer:
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        Args:
            metadata (MetadataCatalog): image metadata.
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode

    def _convert_boxes(self, boxes):
        """
        Convert different format of boxes to an NxB array, where B = 4 or 5 is the box dimension.
        """
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            return boxes.tensor.numpy()
        else:
            return np.asarray(boxes)

    def draw_instance_predictions(self, frame, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            frame (ndarray): an RGB image of shape (H, W, C), in the range [0, 255].
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
            # mask IOU is not yet enabled
            # masks_rles = mask_util.encode(np.asarray(masks.permute(1, 2, 0), order="F"))
            # assert len(masks_rles) == num_instances
        else:
            masks = None

        detected = [
            _DetectedInstance(classes[i], boxes[i], mask_rle=None, color=None, ttl=8)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)

        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        if self._instance_mode == ColorMode.IMAGE_BW:
            # any() returns uint8 tensor
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                (masks.any(dim=0) > 0).numpy() if masks is not None else None
            )
            alpha = 0.3
        else:
            alpha = 0.5

        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            masks=masks,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )

        return frame_visualizer.output

    def next_char(self, word, word_box, boxes, chars, index, box_map, used):
        """
        Args:
            word: string to add new char
            word_box: bbox containing word
            chars: array of chars detected on image
            boxes: array of corresponding bounding boxes
            index: boxes[index] is the position of chars[index] symbol
            box_map: np.ndarray, box_map[i][j]==k means that k-th box includes this pixel
            used: list of indexes of elements that already been used

        Returns: word, word_box, box_map,
        """
        xmin, ymin, xmax, ymax = boxes[index]
        step = max(xmax - xmin, ymax - ymin) // 10
        window = box_map[ymin:ymax, xmin:xmax + step]
        response = window[(window != -1) & (window != index)]
        if not np.size(response):  # if no next char found
            return word, word_box, box_map, used + [index]
        next_index = response[0]
        next_char = chars[next_index]
        nxmin, nymin, nxmax, nymax = boxes[next_index]
        wxmin, wymin, wxmax, wymax = word_box
        word_box = [min(nxmin, wxmin), min(nymin, wymin), max(nxmax, wxmax), max(nymax, wymax)]
        word = word + next_char
        box_map = np.where(box_map == index, -1, box_map)
        return self.next_char(word, word_box, boxes, chars, next_index, box_map, used + [index])

    def merge_chars(self, image, predictions):
        """
        Merges characters into words based on their location
        Args:
            image: numpy array of arbitrary shape with at least 2 dimensions
            predictions: detectron2 prediction objects

        Returns:
            word_boxes: list of lists of 8 ints (polygon corners) and a string - corresponding word
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        boxes = self._convert_boxes(boxes).astype(int)
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        chars = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        chars = np.array([i[0] for i in chars])
        box_map = np.zeros(image.shape[:2]).astype(int)
        box_map.fill(-1)
        total = len(boxes)
        words = []
        chars = chars[boxes[:, 0].argsort()]
        boxes = boxes[boxes[:, 0].argsort()]  # sort by xmin
        used = []
        word_boxes = []
        for i in range(total):
            xmin, ymin, xmax, ymax = boxes[i]
            box_map[ymin:ymax, xmin:xmax] = i
        for i in range(total):
            if i in used:
                continue
            word, word_box, box_map, local_used = self.next_char(chars[i], boxes[i], boxes, chars,
                                                                 i, box_map, [])
            if len(word) < 3:
                continue
            used += local_used
            words.append(word)
            word_boxes.append(word_box)
        return [[*xyxy_abs_to_poly(word_boxes[i]), words[i]] for i in range(len(words))]

    def draw_sem_seg(self, frame, sem_seg, area_threshold=None):
        """
        Args:
            sem_seg (ndarray or Tensor): semantic segmentation of shape (H, W),
                each value is the integer label.
            area_threshold (Optional[int]): only draw segmentations larger than the threshold
        """
        # don't need to do anything special
        frame_visualizer = Visualizer(frame, self.metadata)
        frame_visualizer.draw_sem_seg(sem_seg, area_threshold=None)
        return frame_visualizer.output

    def draw_panoptic_seg_predictions(
            self, frame, panoptic_seg, segments_info, area_threshold=None, alpha=0.5
    ):
        frame_visualizer = Visualizer(frame, self.metadata)
        pred = _PanopticPrediction(panoptic_seg, segments_info, self.metadata)

        if self._instance_mode == ColorMode.IMAGE_BW:
            frame_visualizer.output.img = frame_visualizer._create_grayscale_image(
                pred.non_empty_mask()
            )

        # draw mask for all semantic segments first i.e. "stuff"
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            frame_visualizer.draw_binary_mask(
                mask,
                color=mask_color,
                text=self.metadata.stuff_classes[category_idx],
                alpha=alpha,
                area_threshold=area_threshold,
            )

        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return frame_visualizer.output
        # draw mask for all instances second
        masks, sinfo = list(zip(*all_instances))
        num_instances = len(masks)
        masks_rles = mask_util.encode(
            np.asarray(np.asarray(masks).transpose(1, 2, 0), dtype=np.uint8, order="F")
        )
        assert len(masks_rles) == num_instances

        category_ids = [x["category_id"] for x in sinfo]
        detected = [
            _DetectedInstance(category_ids[i], bbox=None, mask_rle=masks_rles[i], color=None, ttl=8)
            for i in range(num_instances)
        ]
        colors = self._assign_colors(detected)
        labels = [self.metadata.thing_classes[k] for k in category_ids]

        frame_visualizer.overlay_instances(
            boxes=None,
            masks=masks,
            labels=labels,
            keypoints=None,
            assigned_colors=colors,
            alpha=alpha,
        )
        return frame_visualizer.output

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        if instances[0].bbox is None:
            assert instances[0].mask_rle is not None
            # use mask iou only when box iou is None
            # because box seems good enough
            rles_old = [x.mask_rle for x in self._old_instances]
            rles_new = [x.mask_rle for x in instances]
            ious = mask_util.iou(rles_old, rles_new, is_crowd)
            threshold = 0.5
        else:
            boxes_old = [x.bbox for x in self._old_instances]
            boxes_new = [x.bbox for x in instances]
            ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
            threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)
        self._old_instances = instances[:] + extra_instances
        return [d.color for d in instances]
