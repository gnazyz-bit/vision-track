"""Per-stream crossing line configuration, triggers and visual overlays."""
import cv2
import supervision as sv


def counting_line(name, width, height):
    """Preserve the calibrated Crosswalk and Mall lines from the original demo."""
    if name == "Crosswalk":
        y = int(height * 0.51)
        return sv.Point(0, y), sv.Point(int(width * 0.58), y)
    y = int(height * 0.93)
    return sv.Point(int(width * 0.03), y), sv.Point(int(width * 0.91), y)


def make_counting_zone(start, end):
    return sv.LineZone(
        start=start,
        end=end,
        triggering_anchors=[sv.Position.BOTTOM_CENTER]
    )


def update_line_count(zone, detections):
    """Update persistent IN/OUT counts using tracked detections."""
    zone.trigger(detections)


def draw_counting_line(frame, start, end):
    cv2.line(
        frame,
        (int(start.x), int(start.y)),
        (int(end.x), int(end.y)),
        (0, 0, 255),
        3
    )
    return frame