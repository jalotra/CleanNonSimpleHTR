import google_augmentations as augmentations
import numpy as np
from PIL import Image
import cv2
import os

# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(img):
    """Normalize input image channel-wise to zero mean and unit variance."""
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    # print(pil_img.size)
    return np.asarray(pil_img)


def show_image(img):
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    img = cv2.resize(img, (500, 540))      
    cv2.imshow("output", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def augment_and_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: Raw input image as float32 np.ndarray of shape (h, w, c)
        severity: Severity of underlying augmentation operators (between 1 to 10).
        width: Width of augmentation chain
        depth: Depth of augmentation chain. -1 enables stochastic depth uniformly
        from [1, 3]
        alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
        mixed: Augmented and mixed image.
    """
    ws = np.float32(
        np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))

    mix = np.zeros_like(image)
    # print(mix.size, "WS", ws.size)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations.augmentations)
            print(op)
            # op = augmentations.rotate
            image_aug = apply_op(image_aug, op, severity)
    # Preprocessing commutes since all coefficients are convex
            mix += ws[i]*normalize(image_aug)
    # mix = ws[0]*image_aug
    # print(op)
    # show_image(mix)

    mixed = (1 - m) * normalize(image) + m * mix
    show_image(mixed.astype("uint8"))
    return normalize(mixed)

if __name__ == "__main__":
    img = cv2.imread("../toValidate/Box9.jpg", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    img = normalize(img)
    img = augment_and_mix(img)
    print(img, type(img))