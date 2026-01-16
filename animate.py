import imageio.v2 as imageio
import os

# Directory with your exported frames
frame_dir = "/mnt/c/Users/sirpl/Projects/Butterfly Ranges/vanessa-cardui-animation/"

# Get all PNG files sorted by name
frames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])

# Create GIF
images = [imageio.imread(frame) for frame in frames]
imageio.mimsave('vanessa_cardui_animation.gif', images, duration=0.1)  # 0.5 seconds per frame