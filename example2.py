import cv2
import numpy as np
from face_swap import swap_faces

imgPath = "assets/img3.jpg"

# Read images
img = cv2.imread(imgPath)

# Swap faces
img_changed_face = swap_faces(img)
#img_changed_face, process = face_swap.swap_faces2(img) # ignore the process

# Show final result and process
cv2.imshow("result", np.hstack((img, img_changed_face)))
cv2.waitKey(0)

# Save Results
# cv2.imwrite('results/result81.png', result)
# cv2.imwrite('results/process81.png', process)