from matplotlib import pyplot as plt


def plot_torch_image(img, **kwargs):
    '''
    Input image is a torch tensor with the following dims (C,H,W)
    To plot it with matplotlib, we need to change it to (H,W,C)
    kwargs varaible is used to pass other parameters to 'imshow' function.
    '''
    plt.imshow(img.permute(1, 2, 0), **kwargs)
    plt.show()


def plot_losses(plot_name, losses):
    x = [i for i in range(1, len(losses) + 1)]
    plt.plot(x, losses, color='red', linestyle='dashed', linewidth=1, markersize=12)
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.title(plot_name)
    plt.show()
