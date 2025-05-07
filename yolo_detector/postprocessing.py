"""
Post-processing functions for YOLO model predictions.
"""

def apply_post_processing_rules(results):
    """
    Apply rule‑based tweaks to raw YOLO detections for manga bubble layouts.

    Parameters
    ----------
    results : list[ultralytics.engine.results.Results]
        Output from ``model(image_path)``.

    Returns
    -------
    list[list[dict]]
        Cleaned detections per image.
    """
    all_processed = []

    for result in results:
        processed_results = []
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        classes = result.boxes.cls

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box.tolist()
            conf = conf.item()
            cls = int(cls.item())

            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height else 0

            # Rule 1: almost‑square speech bubble → narration
            if 0.9 < aspect_ratio < 1.1 and cls == 0 and conf < 0.9:
                cls = 1

            # Rule 2: extra‑wide rectangle → UI element
            if width / height > 3.0 and cls != 3 and conf < 0.85:
                cls = 3

            processed_results.append({
                'x': x1,
                'y': y1,
                'width': width,
                'height': height,
                'confidence': conf,
                'class': cls
            })
        all_processed.append(processed_results)
    return all_processed