import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_bags(
    bags: np.ndarray,
    labels: np.ndarray,
    idx: int, 
    positive_num: int, 
    show: bool
):
    """
    Visualizes a bag of images and labels for multiple-instance learning (MIL).

    Displays images from a selected bag with individual labels, 
    indicating the overall bag status (positive or negative) based on the target label.

    Args:
        bags (np.ndarray): Array containing images in the bag; shape should be (1, num_images, height, width).
        labels (np.ndarray): Array with bag and instance labels; labels[0] is the bag label, labels[1] holds instance labels.
        idx (int): Index of the bag, used for file naming.
        positive_num (int): Target label considered positive for the bag.
        show (bool): If True, displays the plot; otherwise, saves it to file.

    Saves:
        Plot of bag images with labeled status in "./logs/misc/data" as "bag_{idx}_{bag_status}.png".

    """
    num_images = bags.shape[1]
    num_columns = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_columns))
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))

    axes = axes.flatten()
    is_positive_bag = labels[0] == 1
    bag_status = "Positive" if is_positive_bag else "Negative"
    color = 'green' if is_positive_bag else 'red'
    fig.suptitle(f'Bag Status: {bag_status}', fontsize=14, color=color)
    fig.text(0.5, 0.91, f"positive label: {positive_num}", ha='center', fontsize=12, color='gray')

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(bags[0][i].squeeze(), cmap='gray')

        label_value = labels[1][0][i]
        title_color = 'green' if label_value else 'red'

        ax.set_title(f'Label: {label_value}', color=title_color)
        ax.axis('off')

    for j in range(num_images, num_rows * num_columns):
        axes[j].axis('off')

    plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    else:
        plot_path = os.path.join("./logs/misc/data", f"bag_{idx}_{bag_status}.png")
        try:
            plt.savefig(plot_path)
        except Exception as e:
            print(f"Error saving plot: {e}")
