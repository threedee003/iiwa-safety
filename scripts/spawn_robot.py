from workspace.iiwa_ws import iiwaScene
from workspace.iiwa_shelf_env import ShelfEnv
from isaacgym import gymapi
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2





def run():

      scene = ShelfEnv()
      t = 0
      while scene.viewer_running():
            scene.step()
            x1, d1 = scene.get_rgbd(scene.camera_handles[1])
            x0, d0 = scene.get_rgbd(scene.camera_handles[0])
            seg_map1 = scene.get_segmentation_map(scene.camera_handles[1])
            t0, t1 = scene.camera_transforms[0], scene.camera_transforms[1]
            seg_map0 = scene.get_segmentation_map(scene.camera_handles[0])
            # scene.show_seg_fast(seg_map)
            t += 1
            # if t == 10:
            #       # scene.integrate_tsdf()
            #       # scene.save_tsdf_mesh()
            #       # break
            #       scene.sdf.integrate(d1, seg_map1, t1)
            #       scene.sdf.integrate(d0, seg_map0, t0)
            # if t == 20:
            #       scene.sdf.save_esdf_slices()
            #       # print(np.unique(seg_map))
            # if t > 10.:

            #       Image.fromarray(x[:,:,:3], mode="RGB").save('left_cam.png')
            #       plt.imshow(d, cmap='jet', vmin=0, vmax=3.0)  # 3m max range
            #       plt.colorbar(label="Depth (m)")
            #       plt.axis('off')
            #       plt.savefig("depth_heatmap.png", dpi=300, bbox_inches='tight', pad_inches=0)
            #       plt.close()
            #       break
      # scene.visualise_mesh()
      scene.__del__()





