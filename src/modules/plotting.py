import os
import numpy as np
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def visualize_gtbags(
    bags: np.ndarray,
    labels: np.ndarray,
    idx: int, 
    positive_num: int, 
    show: bool,
    misc_save_path: str
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
    # 0.92 for mu10, 0.94 for mu50, 0.96 for mu100
    fig.text(0.5, 0.96, f"positive label: {positive_num}", ha='center', fontsize=12, color='black')

    for i in range(num_images):
        ax = axes[i]
        ax.imshow(bags[0][i].squeeze().cpu(), cmap='gray')

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
        plot_path = os.path.join(misc_save_path, f"sample_bag_{idx}_{bag_status}.png")
        try:
            plt.savefig(plot_path)
            image = wandb.Image(plot_path, caption="sample bag visualization")
            wandb.log({"sample bag visualization": image})
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close(fig) 

def visualize_attMechanism(
        model, 
        batch: tuple,
        positive_num: int,
        global_step: int,
        show: bool,
        misc_save_path: str
):
    bag, label = batch[0], batch[1]
    y_bag_true = label[0].float()

    y_bag_pred, y_instance_pred = model(bag.squeeze(0))
    if y_instance_pred is None:
        return
    else:
        num_images = batch[0].shape[1]
        num_columns = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_columns))
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 2, num_rows * 2))

        axes = axes.flatten()
        is_positive_bag = label[0] == 1
        bag_status = "Positive" if is_positive_bag else "Negative"
        color = 'green' if is_positive_bag else 'red'
        fig.suptitle(f'Bag Status: {bag_status}', fontsize=14, color=color)
        # 0.92 for mu10, 0.94 for mu50, 0.96 for mu100
        fig.text(0.5, 0.96, f"positive label: {positive_num}", ha='center', fontsize=12, color='black')

        for i in range(num_images):
            ax = axes[i]
            ax.imshow(bag[0][i].squeeze().cpu(), cmap='gray')

            label_value = round(y_instance_pred[i].item(), 5)
            color_v = label[1][0][i]
            title_color = 'green' if color_v else 'red'
    
            ax.set_title(f'a{i} = {label_value}', color=title_color)
            ax.axis('off')

        for j in range(num_images, num_rows * num_columns):
            axes[j].axis('off')

        plt.tight_layout()
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if show:
            plt.show()
        else:
            plot_path = os.path.join(misc_save_path, f"att_bag_{bag_status}.png")
            try:
                plt.savefig(plot_path)
                image = wandb.Image(plot_path, caption="attention mechanism visualization")
                wandb.log({"attention mechanism visualization": image})
            except Exception as e:
                print(f"Error saving plot: {e}")
            finally:
                plt.close(fig) 


def visualize_auc_results(
    mean_bag_size: int,
    var_bag_size: float,
    save_path: str,
    svg_flag: bool
):
    """
    Visualizes AUC validation metric on MNIST-bags for different approaches, like in Figures 1-3 of the paper.
    The approaches are: ['instance+MAX', 'instance+MEAN', 'embedding+MAX', 'embedding+MEAN', 'Attention', 'Gated-Attention']
    Assumes that the results files are complete (include both mean and std) and there is a file corresponding to each case.
    
    Args:
        mean_bag_size (int): The average number of instances per bag, e.g. 10, 50 or 100.
        var_bag_size (float): The variance of the number of instances per bag, e.g. 2, 10, 20.
        save_path (str): The path where the resulting plot will be saved, e.g. './logs/misc/results'.
        svg_flag (bool): Flag stating, whether to save a plot in a vectorized format (.svg) or not (.png).

    Saves:
        Plot comparing AUC for different approaches in the given 'save_path' as 'auc_comparison_{mean_bag_size}'.
    """
    approaches = ['instance_poolmax', 'instance_poolmean', 'embedding_poolmax', 'embedding_poolmean', 'embedding_poolattention', 'embedding_poolgated_attention']
    num_train_bags = [50, 100, 150, 200, 300, 400, 500]
    auc_results = {}

    for approach in approaches:
        approach_results = []
        for num_bags in num_train_bags:
            path_to_res = f"./logs/local_gpu/mu{mean_bag_size}/{approach}_mu{mean_bag_size}_var{var_bag_size}_num{num_bags}/misc/metric_5runs.txt"
            auc_mean = -1
            auc_std = -1
            try:
                with open(path_to_res, 'r') as file:
                    for line in file:
                        if 'Mean' in line:
                            parts = line.split(" ")
                            try:
                                auc_mean = float(parts[1].strip())
                                continue
                            except ValueError:
                                print(f"Error: Convertion of mean '{parts[1]} to float failed.'")
                        elif 'Std' in line:
                            parts = line.split(" ")
                            try:
                                auc_std = float(parts[1].strip())
                                continue
                            except ValueError:
                                print(f"Error: Convertion of std '{parts[1]} to float failed.'")
            except FileNotFoundError:
                print(f"Error: The AUC result file '{path_to_res}' was not found.")
            approach_results.append((auc_mean, auc_std))
        auc_results[approach] = approach_results
    
    colors = ['#1a661a', '#81c784', 'royalblue', 'skyblue', 'salmon', 'firebrick']
    markers = ['o', 's', '^', '*', 'D', 'v']
    labels = ['instance+MAX', 'instance+MEAN', 'embedding+MAX', 'embedding+MEAN', 'Attention', 'Gated-Attention']

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (approach, data) in enumerate(auc_results.items()):
        means = [point[0] for point in data]
        stds = [point[1] for point in data]
        ax.errorbar(
            num_train_bags, means, yerr=stds, label=labels[idx], color=colors[idx],
            marker=markers[idx], linestyle='dashdot', capsize=5
        )

    ax.set_xlabel('Number of training bags')
    ax.set_ylabel('AUC')
    ax.set_ylim((0.55, 1.0))
    loc = 'center right' if mean_bag_size == 10 else 'lower right'
    ax.legend(loc=loc)
    ax.grid(which='major', color='gray', linestyle='-', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3)
    file_format = 'svg' if svg_flag else 'png'
    fig.savefig(f"{save_path}/auc_results_{mean_bag_size}.{file_format}", bbox_inches='tight')

### test
visualize_auc_results(10, 2, "./logs", False)
visualize_auc_results(50, 10, "./logs", False)
