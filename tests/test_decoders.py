import numpy as np
from multiarch_onnx import detect_contract, decode_outputs


def test_detect_contracts():
    assert detect_contract(["dets", "labels"], [(1, 300, 4), (1, 300, 5)]) == "rfdetr"
    assert detect_contract(["logits", "boxes"], [(1, 300, 49), (1, 300, 4)]) == "dfine"
    assert detect_contract(["output0"], [(1, 300, 53)]) == "rtdetr"
    assert detect_contract(["output0"], [(1, 84, 8400)]) == "yolo"


def test_decode_dfine_nms_free():
    logits = np.full((1, 300, 4), -5.0, dtype=np.float32)
    logits[0, 0, 2] = 3.0
    boxes = np.zeros((1, 300, 4), dtype=np.float32)
    boxes[0, 0] = [0.5, 0.5, 0.2, 0.4]
    dets = decode_outputs([logits, boxes], ["logits", "boxes"], img_w=640, img_h=640, conf_thresh=0.5)
    assert len(dets) == 1
    assert dets[0]["class_id"] == 2
    assert dets[0]["score"] > 0.9


def test_decode_rfdetr_c_plus_1():
    dets_raw = np.zeros((1, 300, 4), dtype=np.float32)
    dets_raw[0, 1] = [0.5, 0.5, 0.25, 0.25]
    labels = np.full((1, 300, 5), -6.0, dtype=np.float32)
    labels[0, 1, 1] = 2.5
    labels[0, 1, 4] = 4.0
    res = decode_outputs([dets_raw, labels], ["dets", "labels"], img_w=560, img_h=560, conf_thresh=0.5, num_classes=4)
    assert len(res) == 1
    assert res[0]["class_id"] == 1


def test_decode_yolo_standard_and_nms():
    raw = np.zeros((1, 6, 10), dtype=np.float32)
    raw[0, :, 0] = [100, 100, 40, 40, 0.92, 0.05]
    raw[0, :, 1] = [102, 101, 40, 40, 0.80, 0.05]
    raw[0, :, 2] = [400, 400, 50, 50, 0.02, 0.88]
    dets = decode_outputs([raw], ["output0"], img_w=640, img_h=640, conf_thresh=0.5, iou_thresh=0.45)
    assert len(dets) == 2
    assert {d["class_id"] for d in dets} == {0, 1}
