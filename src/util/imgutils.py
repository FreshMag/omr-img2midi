import warnings

import cv2
import matplotlib.pyplot as plt


def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def show_grid(images, grid_size=(3, 3)):
    """
    Display a grid of images.

    Parameters:
        images (list of numpy.ndarray): List of images to display.
        grid_size (tuple): Number of rows and columns in the grid.
    """
    num_images = len(images)
    rows, cols = grid_size
    total_subplots = rows * cols

    if num_images > total_subplots:
        warnings.warn("Warning: Not all images will be displayed. Grid size is too small.")

    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
