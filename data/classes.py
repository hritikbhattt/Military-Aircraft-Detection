"""
Canonical class list for the 36 supported aircraft.

CLASSES is the order used everywhere (YOLO class indices == list index).
RAW_TO_CLASS maps the short codes used in the source Kaggle dataset's
annotation CSVs (e.g. "F16", "SU57") to our display names (e.g. "F-16", "Su-57").
Extend RAW_TO_CLASS if you add more source aircraft codes later.
"""

CLASSES = [
    "F-22", "F-35", "F-16", "F-15", "F-18", "F-14", "F-4",
    "B-2", "B-1", "B-52", "F-117", "SR-71",
    "A-10", "C-130", "C-17", "C-5", "U-2",
    "YF-23", "XB-70",
    "Su-57", "MiG-31", "Tu-95", "Tu-160", "J-20",
    "Rafale", "EF2000", "JAS-39", "Mirage-2000",
    "V-22", "MQ-9", "RQ-4", "E-2",
    "AG600", "Be200", "US-2", "A400M",
]

assert len(CLASSES) == 36

# Maps raw label strings found in the source dataset's CSV annotations
# to the canonical names above. Keys are upper-cased/underscore-stripped
# at lookup time, so "f16", "F16", "F-16" all resolve the same way.
RAW_TO_CLASS = {
    "F22": "F-22", "F35": "F-35", "F16": "F-16", "F15": "F-15",
    "F18": "F-18", "F14": "F-14", "F4": "F-4",
    "B2": "B-2", "B1": "B-1", "B52": "B-52",
    "F117": "F-117", "SR71": "SR-71", "SR71A12": "SR-71",
    "A10": "A-10", "C130": "C-130", "C17": "C-17", "C5": "C-5", "U2": "U-2",
    "YF23": "YF-23", "XB70": "XB-70",
    "SU57": "Su-57", "MIG31": "MiG-31", "TU95": "Tu-95",
    "TU142": "Tu-95", "TU160": "Tu-160", "J20": "J-20",
    "RAFALE": "Rafale", "EF2000": "EF2000", "JAS39": "JAS-39",
    "MIRAGE2000": "Mirage-2000",
    "V22": "V-22", "MQ9": "MQ-9", "RQ4": "RQ-4", "E2": "E-2",
    "AG600": "AG600", "BE200": "Be200", "US2": "US-2", "A400M": "A400M",
}


def normalize_raw_label(raw: str) -> str | None:
    """Return the canonical class name for a raw dataset label, or None if unsupported."""
    key = raw.strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    return RAW_TO_CLASS.get(key)


def class_index(name: str) -> int:
    return CLASSES.index(name)
