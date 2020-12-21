from torchvision.transforms import transforms


def augment_images(images, operators=None):
    augmentor = transforms.Compose([transforms.RandomAffine(degrees=20)])
    images_augmented = augmentor(images)
    return images_augmented
