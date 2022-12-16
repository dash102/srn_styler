import cv2
import os

style = 'blue'

for dir_int in range(10):
    dir = f'{dir_int:06}'
    for rendering_type in {'gt_comparisons', 'renderings'}:
        image_folder = f'/data/vision/phillipi/fus/srn/srn_logs/{style}/val/{rendering_type}/{dir}'
        video_name = f'videos/{style}_{dir}_{rendering_type}.avi'

        images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 1, (width,height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        # cv2.destroyAllWindows()
        video.release()