import cv2
import numpy as np
from image_utils import resize_and_border_image, concatenate_images
from face_swap import swap_faces


imgPath = "assets/img1.jpg"
imgPath2 = "assets/img2.jpg"

# Read images
img = cv2.imread(imgPath)
img2 = cv2.imread(imgPath2)

# Resize and border image, so both images have same dimensions
img, img2 = resize_and_border_image(img, img2)

# Swap faces
img_changed_face, img2_changed_face = swap_faces(img, img2)
#img_changed_face, img2_changed_face = face_swap.swap_faces(img, img2) # ignore the process

# Concatenate results
result = concatenate_images(
    (
        concatenate_images((img, img_changed_face), border=10),
        concatenate_images((img2, img2_changed_face), border=10)
    ),
    axis='v',
    border=10
)
# Resize results so they fit in the screen
result = cv2.resize(result, (int(result.shape[1] / 3), int(result.shape[0] / 3)))

# Show results
cv2.imshow("result", result)
cv2.waitKey(0)

# Save Results
# cv2.imwrite('results/result81.png', result)
# cv2.imwrite('results/process81.png', process)