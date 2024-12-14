import numpy as np
from ctra_bytetrack import CTRAByteTracker
from basetrack import TrackState

# Generate synthetic test data
def generate_test_data(num_frames, num_objects):
    """
    Generate test data for object tracking.
    Each object moves in a straight line with some noise added.
    """
    np.random.seed(42)
    detections = []
    for frame in range(num_frames):
        frame_detections = []
        for obj_id in range(num_objects):
            x = obj_id * 10 + frame * 2 + np.random.normal(0, 1)
            y = obj_id * 5 + frame * 1 + np.random.normal(0, 1)
            bbox = [x, y, 2, 2, 0.9]  # x, y, w, h, score
            frame_detections.append(bbox)
        detections.append(np.array(frame_detections))
    return detections

# Test setup
def test_ctra_tracker():
    # Parameters
    num_frames = 20
    num_objects = 5
    frame_rate = 30

    # Generate synthetic detections
    detections = generate_test_data(num_frames, num_objects)

    # Initialize tracker
    class Args:
        track_thresh = 0.5
        match_thresh = 0.7
        track_buffer = 30
        mot20 = False
    
    args = Args()
    tracker = CTRAByteTracker(args, frame_rate=frame_rate)

    # Run tracker on each frame
    all_tracks = []
    for frame_id, frame_detections in enumerate(detections, 1):
        img_info = (720, 1280)  # Example image dimensions
        img_size = (720, 1280)
        tracked_objects = tracker.update(frame_detections, img_info, img_size)

        # Store and display tracking results
        frame_tracks = []
        for track in tracked_objects:
            if track.state == TrackState.Tracked:
                frame_tracks.append({
                    "id": track.track_id,
                    "bbox": track.tlbr,
                    "frame": frame_id
                })
        all_tracks.append(frame_tracks)

    return all_tracks

# Visualize results
def visualize_tracks(tracks, num_frames, num_objects):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    colors = plt.cm.jet(np.linspace(0, 1, num_objects))
    for frame_tracks in tracks:
        for track in frame_tracks:
            track_id = track["id"]
            bbox = track["bbox"]
            plt.scatter(bbox[0], bbox[1], color=colors[track_id % num_objects], label=f"Track {track_id}")
    
    plt.title("CTRA Tracker Results")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid()
    plt.show()

# Run the test
if __name__ == "__main__":
    tracks = test_ctra_tracker()
    visualize_tracks(tracks, num_frames=20, num_objects=5)
