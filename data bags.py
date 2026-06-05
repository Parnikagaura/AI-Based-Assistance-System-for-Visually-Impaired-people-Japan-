import pyrealsense2 as rs
import numpy as np
import cv2
import os
import glob
import csv

# ================= SETTINGS ================= #
input_folder = "./Rgb camera data zebra crossings/data_zebra_crossing"          # folder containing multiple .bag files
output_root = "./dataset"

os.makedirs(output_root, exist_ok=True)

# ============================================ #

def resize_and_crop(color, depth):
    h, w = color.shape[:2]

    # Resize to height 512(keep aspect ratio)
    new_h = 512
    scale = new_h / h
    new_w = int(w * scale)

    color_resized = cv2.resize(color, (new_w, new_h))
    depth_resized = cv2.resize(depth, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)

    # Center crop 512 * 256
    crop_w = 256zss
    center_x = new_w // 2
    start_x = center_x - crop_w // 2
    end_x = start_x + crop_w

    color_crop = color_resized[:, start_x:end_x]
    depth_crop = depth_resized[:, start_x:end_x]

    return color_crop, depth_crop


def process_bag(bag_path, dataset_index):

    print(f"Processing: {bag_path}")

    # Create dataset folder (1,2,3,...)
    dataset_folder = os.path.join(output_root, str(dataset_index))
    rgb_folder = os.path.join(dataset_folder, "RGB")
    depth_folder = os.path.join(dataset_folder, "Depth")
    imu_folder = os.path.join(dataset_folder, "IMU")

    os.makedirs(rgb_folder, exist_ok=True)
    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(imu_folder, exist_ok=True)

    # Setup pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(bag_path, repeat_playback=False)

    pipeline.start(config)

    profile = pipeline.get_active_profile()
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    align = rs.align(rs.stream.color)

    frame_index = 0
    saved_index = 0

    imu_data = []

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # IMU streams
            for frame in frames:
                if frame.is_motion_frame():
                    motion = frame.as_motion_frame().get_motion_data()
                    imu_data.append([
                        frame.get_timestamp(),
                        motion.x,
                        motion.y,
                        motion.z
                    ])

            if not color_frame or not depth_frame:
                continue

            frame_index += 1

            # Keep only 10th, 20th, 30th frame per second (30 FPS)
            frame_mod = frame_index % 30
            if frame_mod not in [10, 20, 0]:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            color_crop, depth_crop = resize_and_crop(color_image, depth_image)

            # Save RGB
            rgb_path = os.path.join(rgb_folder, f"{saved_index:06d}.png")
            cv2.imwrite(rgb_path, color_crop)

            # Save Depth
            depth_path = os.path.join(depth_folder, f"{saved_index:06d}.npy")
            np.save(depth_path, depth_crop)

            saved_index += 1

    except RuntimeError:
        print("Finished:", bag_path)

    finally:
        pipeline.stop()

    # Save IMU as CSV
    imu_csv_path = os.path.join(imu_folder, "imu_data.csv")
    with open(imu_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y", "z"])
        writer.writerows(imu_data)


# ============== PROCESS ALL BAG FILES ============== #

bag_files = sorted(glob.glob(os.path.join(input_folder, "*.bag")))

for i, bag_file in enumerate(bag_files, start=1):
    process_bag(bag_file, i)

print("All files processed successfully.")
