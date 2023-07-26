import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
    ]

    class_order = np.arange(10).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = []
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    ]

    class_order = np.arange(100).tolist()

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100('./data', train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100('./data', train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)

class iCIFAR100_224(iCIFAR100):
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]

class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=63/255)
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/imagenet100/train/'
        test_dir = 'data/imagenet100/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetR(iData):
    use_path = True
    train_trsf = [
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/imagenet-r/train/'
        test_dir = 'data/imagenet-r/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCUB200_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/cub_200/train/'
        test_dir = 'data/cub_200/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iCARS196_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/cars196/train/'
        test_dir = 'data/cars196/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iResisc45_224(iData): 
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/resisc45/train/'
        test_dir = 'data/resisc45/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)



class iSketch345_224(iData):
    use_path = True
    train_trsf = [
        transforms.Resize((300, 300), interpolation=3),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        train_dir = 'data/sketch345/train/'
        test_dir = 'data/sketch345/val/'

        train_dset = datasets.ImageFolder(train_dir)
        test_dset = datasets.ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)