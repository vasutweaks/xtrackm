import os

selected_tuples = (
    ("TN", "ERS1+ERS2+ENV+SRL", "23"),
    ("TN", "ERS1+ERS2+ENV+SRL", "868"),
    ("TN", "S3A", "137"),
    ("TN", "S3A", "608"),
    ("TN", "S3B", "137"),
    ("TN", "TP+J1+J2+J3+S6A", "155"),
    ("TN", "TPN+J1N+J2N+J3N", "40"),
    ("AP", "S3A", "608"),
    ("AP", "S3A", "722"),
    ("AP", "TP+J1+J2+J3+S6A", "155"),
    ("AP", "TPN+J1N+J2N+J3N", "155"),
    ("OR", "ERS1+ERS2+ENV+SRL", "137"),
    ("OR", "ERS1+ERS2+ENV+SRL", "595"),
    ("OR", "ERS1+ERS2+ENV+SRL", "610"),
    ("OR", "HY2A", "95"),
    ("OR", "S3A", "165"),
    ("OR", "S3A", "180"),
    ("OR", "S3A", "294"),
    ("OR", "TPN+J1N+J2N+J3N", "192"),
)

for radar_str, sat, track_number in selected_tuples:
    cmd = f"python3 p.radar_rotatedVStrack_gc_c.ek.along.intersections.selected.py --radar-str {radar_str} --sat {sat} --track-number {track_number}"
    os.system(cmd)
