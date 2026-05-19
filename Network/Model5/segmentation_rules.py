import config


def _matches_any(name, patterns):
    lower = name.lower()
    return any(pattern in lower for pattern in patterns)


class SegmentationRules:
    """Segmentation IDs and object-name rules shared by Model5's A* mapper."""

    def __init__(self, client):
        self.client = client

    def apply(self):
        """Apply the same high-level convention used by Model2."""
        print("[Model5] Setting segmentation IDs...")
        self.client.simSetSegmentationObjectID(".*", 0, True)

        for pattern in config.FREE_OBJECT_PATTERNS:
            self.client.simSetSegmentationObjectID(f".*{pattern}.*", 1, True)

        for pattern in config.OBSTACLE_OBJECT_PATTERNS:
            self.client.simSetSegmentationObjectID(f".*{pattern}.*", 2, True)

        # Keep Model2's explicit names as a small compatibility anchor.
        self.client.simSetSegmentationObjectID("SM_AM_vol8_sidewalk.*", 1, True)
        self.client.simSetSegmentationObjectID("SM_AM_vol8_street.*", 1, True)
        self.client.simSetSegmentationObjectID("SM_AM_vol8_curb.*", 1, True)
        self.client.simSetSegmentationObjectID("AM_vol8_building.*", 2, True)
        print("[Model5] Segmentation IDs set: free=1, obstacle=2")

    def classify_name(self, object_name):
        if _matches_any(object_name, config.OBSTACLE_OBJECT_PATTERNS):
            return "obstacle"
        if _matches_any(object_name, config.FREE_OBJECT_PATTERNS):
            return "free"
        return "unknown"

    def segmentation_id(self, object_name):
        try:
            return int(self.client.simGetSegmentationObjectID(object_name))
        except Exception:
            return None

    def is_blocked_object(self, object_name):
        seg_id = self.segmentation_id(object_name)
        if seg_id in config.BLOCKED_SEGMENTATION_IDS:
            return True
        if seg_id in config.FREE_SEGMENTATION_IDS:
            return False
        return self.classify_name(object_name) == "obstacle"

    def obstacle_objects(self):
        """
        Return likely obstacle objects without querying every scene object's
        segmentation ID.  Large UE scenes can contain thousands of meshes, and
        simGetSegmentationObjectID per mesh is painfully slow.
        """
        blocked = set()
        for pattern in config.OBSTACLE_OBJECT_PATTERNS:
            try:
                names = self.client.simListSceneObjects(f".*{pattern}.*")
            except Exception:
                names = []
            blocked.update(names)
        return sorted(blocked)
