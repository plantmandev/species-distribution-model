import imageio.v2 as imageio
import os

frame_dir = "/mnt/c/Users/sirpl/Projects/species-distribution-model/visualization/danaus-plexippus-input"
frames = sorted(
    os.path.join(frame_dir, f)
    for f in os.listdir(frame_dir)
    if f.endswith(".png")
)

# Output MP4 file
output_file = "danaus-plexippus-animation.mp4"

# Write frames to MP4
with imageio.get_writer(output_file, fps=24) as writer:
    for frame_path in frames:
        writer.append_data(imageio.imread(frame_path))

print(f"Finished writing {output_file}")