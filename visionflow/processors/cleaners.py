import numpy as np
from PIL import Image

def remove_black_border(image: Image.Image) -> Image.Image:
    img_array_orig = np.array(image)
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = np.array(image.convert('RGBA'))[:, :, 3]
        non_black = alpha > 0 if np.any(alpha < 255) else np.array(image.convert('L')) > 10
    else:
        non_black = np.array(image.convert('L')) > 10

    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)

    if not np.any(rows) or not np.any(cols):
        return image.convert('RGB')

    top, bottom = np.argmax(rows), len(rows) - np.argmax(rows[::-1])
    left, right = np.argmax(cols), len(cols) - np.argmax(cols[::-1])

    if bottom <= top: bottom = top + 1
    if right <= left: right = left + 1

    try:
        cropped = img_array_orig[top:bottom, left:right]
        return Image.fromarray(cropped).convert('RGB')
    except IndexError:
        return image.convert('RGB')