import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
from imgaug import augmenters as iaa
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
x_train_file_npy = ".//input//x_train.npy"
y_train_file_npy = ".//input//y_train.npy"
x_test_file_npy = ".//input//x_test.npy"


def augment_images(x_train=None, y_train=None, augs=2):
    # images must be provided in 28,28,1 and unit8, labels as uint8
    print('Augmenting images...')
    if augs <= 1:
        print('multiplier must be 2 or greater. Aborting.')
        return x_train, y_train

    file_x_train = f'.//x_train_{str(augs)}x.npy'
    file_y_train = f'.//y_train_{str(augs)}x.npy'
    x_train_inflated = x_train
    y_train_inflated = y_train
    for x in range(0, augs - 1):
        x_train_inflated = np.concatenate((x_train_inflated, x_train), axis=0)
        y_train_inflated = np.concatenate((y_train_inflated, y_train), axis=0)
    print(f'Augmenting: {x_train.shape} to : {x_train_inflated.shape}')

    # randomly augment the images
    seq = iaa.Sequential(
        [
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.Affine(rotate=(0.5, 1.5)),
            iaa.CropAndPad(percent=(-0.25, 0.25)),
            # iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))

        ]
    )
    #
    # seq = iaa.OneOf([
    #                    iaa.Sequential(
    #                         [
    #                             iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    #                             iaa.Affine(rotate=(0.5, 1.5)),
    #                             iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    #                             iaa.CropAndPad(percent=(-0.25, 0.25))
    #                         ]
    #                     ),
    #                     iaa.Sequential(
    #                         [
    #                             iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    #                             iaa.Affine(rotate=(0.5, 1.5)),
    #                             iaa.CropAndPad(percent=(-0.25, 0.25)),
    #                             iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))
    #                         ]
    #                     ),
    #                     iaa.Sequential(
    #                         [
    #                             iaa.CoarseDropout(0.10, size_percent=0.33),
    #                             iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
    #                             iaa.Affine(rotate=(0.5, 1.5)),
    #                             iaa.CropAndPad(percent=(-0.25, 0.25))
    #                         ]
    #                     )
    #             ])

    x_train_aug = seq.augment_images(x_train_inflated)
    # insert the original 60k images and labels at the beginning
    # x_train = np.concatenate((x_train, x_train_aug), axis=0)
    # y_train = np.concatenate((y_train, y_train_inflated), axis=0)
    print('Augmention complete.')

    # save to file
    np.save(file_x_train, x_train_aug)
    np.save(file_y_train, y_train_inflated)
    print('Saved.')

    return x_train, y_train


# convert_csv_to_pnp(create_digit_pngs=True)


# ----------------------------------
def gen_augmentations():
    for x in range(0, 1):
        x_train = np.load(file=x_train_file_npy)
        y_train = np.load(file=y_train_file_npy)
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_train, y_train = augment_images(x_train=x_train, y_train=y_train, augs=x)
