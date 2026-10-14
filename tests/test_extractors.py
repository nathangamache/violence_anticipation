import numpy as np
from core.utils import center_of_bbox

def test_center_bbox():
    """
    Verify center_of_bbox returns correct midpoint.
    """
    bbox = [0, 0, 10, 10]
    cxcy = center_of_bbox(bbox)

    assert cxcy[0] == 5
    assert cxcy[1] == 5
    assert cxcy.dtype == np.float32

    print("[PASS] test_center_bbox")


def test_import_extractors():
    """
    Ensure extractor functions import cleanly.
    """
    from core.extractors import (
        detect_and_track,
        extract_actions,
        extract_audio,
        extract_scene,
        live_track,
        live_actions,
        live_scene,
        live_audio
    )

    # Just verifying the names exist
    assert callable(detect_and_track)
    assert callable(extract_actions)
    assert callable(extract_audio)
    assert callable(extract_scene)
    assert callable(live_track)
    assert callable(live_actions)
    assert callable(live_scene)
    assert callable(live_audio)

    print("[PASS] test_import_extractors")
