"""
Canonical class list for the Military Aircraft Detection model.

Source: labels_with_split.csv from the Kaggle "Military Aircraft Detection
Dataset" (a2015003713). These are the exact 103 class labels as they appear
in the dataset's `class` column.
"""

# Sorted list of all 103 aircraft classes in the dataset.
CANONICAL_CLASSES = [
    "A10", "A400M", "AG600", "AH64", "AKINCI", "AV8B", "An124", "An22",
    "An225", "An72", "B1", "B2", "B21", "B52", "Be200", "C1", "C130", "C17",
    "C2", "C390", "C5", "CH47", "CH53", "CL415", "E2", "E7", "EF2000",
    "EMB314", "F117", "F14", "F15", "F16", "F18", "F2", "F22", "F35", "F4",
    "FCK1", "H6", "Il76", "J10", "J20", "J35", "J36", "J50", "JAS39", "JF17",
    "JH7", "KAAN", "KC135", "KF21", "KIZILELMA", "KJ600", "Ka27", "Ka52",
    "MQ20", "MQ25", "MQ28", "MQ35", "MQ9", "Mi24", "Mi26", "Mi28", "Mi8",
    "Mig29", "Mig31", "Mirage2000", "NH90", "P3", "RQ4", "Rafale", "SR71",
    "Su24", "Su25", "Su34", "Su47", "Su57", "T50", "TB001", "TB2", "Tejas",
    "Tornado", "Tu160", "Tu22M", "Tu95", "U2", "UH60", "US2", "V22", "V280",
    "Vulcan", "WZ10", "WZ7", "WZ9", "X29", "X32", "XB70", "XQ58", "Y20",
    "YF23", "Z10", "Z19", "Z21",
]

NUM_CLASSES = len(CANONICAL_CLASSES)

# Maps class name -> index (0-based), used to build YOLO label files.
CLASS_TO_INDEX = {name: i for i, name in enumerate(CANONICAL_CLASSES)}
INDEX_TO_CLASS = {i: name for name, i in CLASS_TO_INDEX.items()}


def normalize_label(raw_label: str) -> str:
    """
    Cleans a raw label string from the CSV (strips whitespace) and validates
    it against the canonical class list. Raises if it's not a known class,
    so bad rows fail loudly instead of silently getting dropped later.
    """
    cleaned = raw_label.strip()
    if cleaned not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown aircraft class label: {raw_label!r}")
    return cleaned