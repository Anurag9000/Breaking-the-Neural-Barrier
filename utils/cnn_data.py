import torchvision.transforms as T

# CIFAR channel stats
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN, CIFAR100_STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)


def make_cifar_transforms(dataset: str, use_augment: bool = True):
    """
    Returns (train_tfms, eval_tfms) for CIFAR-10/100.
    - Augment: RandomCrop(32, padding=4) + RandomHorizontalFlip
    - Always normalizes with dataset-specific mean/std.
    """
    ds = dataset.lower()
    if ds == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif ds == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError("dataset must be 'cifar10' or 'cifar100'")

    aug = []
    if use_augment:
        aug = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]

    train_tfms = T.Compose([*aug, T.ToTensor(), T.Normalize(mean, std)])
    eval_tfms = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_tfms, eval_tfms
