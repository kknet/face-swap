import cv2
import numpy as np
from image_utils import resize_and_border_image, create_gif
from face_swap import get_triangles_img, show_process

def append_triangles_img(img, img2):
    for (i1, i2, img_new_face) in get_triangles_img(img, img2):

        result = np.hstack((i1, i2, img_new_face))
        result = cv2.resize(result, (int(result.shape[1] / 4), int(result.shape[0] / 4)))
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        yield result


imgPath = "assets/img1.jpg"
imgPath2 = "assets/img2.jpg"

# Read images
img = cv2.imread(imgPath)
img2 = cv2.imread(imgPath2)

# Resize and border image, so both images have same dimensions
img, img2 = resize_and_border_image(img, img2)

first_steps = show_process(img, img2)

first_steps = np.vstack(
    (
        np.hstack(first_steps[0]),
        np.hstack(first_steps[1])
    )
)

first_steps = cv2.resize(first_steps, (int(first_steps.shape[1]/3), int(first_steps.shape[0]/3)))
cv2.imwrite('results/first_steps81.png', first_steps)
second_step = create_gif('results/second_step81.gif', append_triangles_img(img, img2))

#cv2.imshow("result", first_steps)
#cv2.waitKey(0)