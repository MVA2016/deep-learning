import matplotlib.pyplot as plt
from scipy.io import loadmat


def load_digits(filename):
    '''
    Load digits for assignment 1

    Parameters
    ----------
    filename: str
        path to .mat file

    Returns
    -------
    tuple(ndarray,ndarray)
        Returns a tuple with train set and test set
    '''
    digit7 = loadmat(filename)['D'].astype(float)/255.
    train = digit7[::2]
    test = digit7[1::2]
    print('Train set {}'.format(train.shape))
    print('Test set {}'.format(test.shape))
    return train, test


def show_images(**kwargs):
    '''
    Plot images with titles side by side

    Example
        show_image(title1=im1, title2=im2)
    '''
    for i, (title, im) in enumerate(kwargs.items()):
        plt.subplot(1, len(kwargs), i+1)
        plt.title(title)
        plt.imshow(im)
    plt.draw()


# Select interactive mode
plt.ion()
