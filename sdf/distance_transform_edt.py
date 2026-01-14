from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



path = "/home/bikram/Documents/isaacgym/iiwa_safety/segmentation.png"

def show_seg_fast(seg_map, legend = False):
      plt.figure(figsize=(8,6))
      plt.imshow(seg_map, cmap="tab20")   # quick categorical colormap
      # plt.title("Segmentation map (IDs)")
      if legend:
            plt.colorbar(label="seg id")
      plt.axis("off")
      plt.show()

if __name__ == '__main__':
      f = Image.open(path)
      f = np.array(f) / 255.
      f = np.abs(1-f)
      show_seg_fast(f)
      
      y = distance_transform_edt(f)
      show_seg_fast(y)