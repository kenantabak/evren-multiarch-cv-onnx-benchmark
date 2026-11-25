from __future__ import annotations
import time
from typing import Any, Dict, List, Tuple
import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.45) -> List[int]:
    """Pure NumPy Greedy NMS for [N, 4] xyxy boxes."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thresh]
    return keep


def detect_contract(output_names: List[str], output_shapes: List[Tuple[Any, ...]]) -> str:
    """Identify ONNX architecture output contract automatically."""
    names = set(output_names)
    if {"dets", "labels"}.issubset(names):
        return "rfdetr"
    if {"logits", "boxes"}.issubset(names):
        return "dfine"
    if len(output_shapes) == 1:
        shp = output_shapes[0]
        if len(shp) == 3 and shp[1] == 300:
            return "rtdetr"
        return "yolo"
    return "unknown"


def _cxcywh_to_xyxy(b: np.ndarray, img_w: int, img_h: int, normalized: bool = True) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    if normalized:
        cx, cy, w, h = cx * img_w, cy * img_h, w * img_w, h * img_h
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)


def decode_outputs(
    outputs: List[np.ndarray],
    output_names: List[str],
    img_w: int,
    img_h: int,
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    num_classes: int | None = None,
) -> List[Dict[str, Any]]:
    """Unified pure-NumPy post-processor for YOLO, RT-DETR, D-FINE, and RF-DETR ONNX models."""
    contract = detect_contract(output_names, [o.shape for o in outputs])
    by_name = dict(zip(output_names, outputs))
    dets: List[Dict[str, Any]] = []

    if contract == "rfdetr":
        boxes_raw = by_name["dets"][0]
        logits = by_name["labels"][0]
        probs = _sigmoid(logits)
        fg_probs = probs[:, :num_classes] if (num_classes and probs.shape[1] > num_classes) else probs
        cls_ids = np.argmax(fg_probs, axis=1)
        scores = fg_probs[np.arange(len(cls_ids)), cls_ids]
        mask = scores >= conf_thresh
        xyxy = _cxcywh_to_xyxy(boxes_raw[mask], img_w, img_h, normalized=True)
        for box, sc, cid in zip(xyxy, scores[mask], cls_ids[mask]):
            dets.append({"xyxy": [float(v) for v in box], "score": float(sc), "class_id": int(cid)})
        return dets

    if contract == "dfine":
        boxes_raw = by_name["boxes"][0]
        logits = by_name["logits"][0]
        probs = _sigmoid(logits)
        cls_ids = np.argmax(probs, axis=1)
        scores = probs[np.arange(len(cls_ids)), cls_ids]
        mask = scores >= conf_thresh
        xyxy = _cxcywh_to_xyxy(boxes_raw[mask], img_w, img_h, normalized=True)
        for box, sc, cid in zip(xyxy, scores[mask], cls_ids[mask]):
            dets.append({"xyxy": [float(v) for v in box], "score": float(sc), "class_id": int(cid)})
        return dets

    if contract == "rtdetr":
        arr = outputs[0][0]
        boxes_raw = arr[:, :4]
        scores_mat = arr[:, 4:]
        if scores_mat.max() > 1.0 or scores_mat.min() < 0.0:
            scores_mat = _sigmoid(scores_mat)
        cls_ids = np.argmax(scores_mat, axis=1)
        scores = scores_mat[np.arange(len(cls_ids)), cls_ids]
        mask = scores >= conf_thresh
        normalized = bool(boxes_raw.max() <= 1.5)
        xyxy = _cxcywh_to_xyxy(boxes_raw[mask], img_w, img_h, normalized=normalized)
        for box, sc, cid in zip(xyxy, scores[mask], cls_ids[mask]):
            dets.append({"xyxy": [float(v) for v in box], "score": float(sc), "class_id": int(cid)})
        return dets

    if contract == "yolo":
        raw = outputs[0][0]
        if raw.ndim == 2 and raw.shape[1] == 6:
            mask = raw[:, 4] >= conf_thresh
            for row in raw[mask]:
                dets.append({"xyxy": [float(v) for v in row[:4]], "score": float(row[4]), "class_id": int(row[5])})
            return dets
        pred = raw.T if raw.shape[0] < raw.shape[1] else raw
        boxes_raw = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        scores = cls_scores[np.arange(len(cls_ids)), cls_ids]
        mask = scores >= conf_thresh
        if not np.any(mask):
            return []
        xyxy = _cxcywh_to_xyxy(boxes_raw[mask], img_w, img_h, normalized=False)
        keep = nms_xyxy(xyxy, scores[mask], iou_thresh=iou_thresh)
        for k in keep:
            dets.append({"xyxy": [float(v) for v in xyxy[k]], "score": float(scores[mask][k]), "class_id": int(cls_ids[mask][k])})
        return dets

    raise ValueError(f"Unsupported ONNX contract: outputs={output_names}")


def benchmark_onnx_cpu(model_path: str, default_imgsz: int = 640, threads: int = 4, warmup: int = 3, runs: int = 10) -> Dict[str, Any]:
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    inps = sess.get_inputs()
    outs = sess.get_outputs()
    feed: Dict[str, np.ndarray] = {}
    for inp in inps:
        shp = [d if isinstance(d, int) and d > 0 else (1 if i == 0 else default_imgsz) for i, d in enumerate(inp.shape)]
        if "int64" in inp.type:
            feed[inp.name] = np.array([[default_imgsz, default_imgsz]], dtype=np.int64)
        else:
            feed[inp.name] = np.random.rand(*shp).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    med = round(float(np.median(times)), 1)
    return {
        "contract": detect_contract([o.name for o in outs], [tuple(o.shape) for o in outs]),
        "inputs": [(i.name, i.shape) for i in inps],
        "outputs": [(o.name, o.shape) for o in outs],
        "median_ms": med,
        "fps": round(1000.0 / med, 2),
    }
