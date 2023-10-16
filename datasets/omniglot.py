import  torch.utils.data as data
import  os
import  os.path
import  errno
from .datasets import register
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import torch
import time
@register('omniglot')
class Omniglot(data.Dataset):
    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    '''
    The items are (filename,category). The index of all the categories can be found in self.idx_classes
    Args:
    - root: the directory where the dataset will be stored
    - transform: how to transform the input
    - target_transform: how to transform the target
    - download: need to download the dataset
    '''

    def __init__(self, root_path, split='train', **kwargs):

        download = False
        #self.n_way = 1200
        root = '/data/Datasets/WBCDATA/Omniglot'
        self.root = root

        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')



        self.data = []
        self.label = []


        if split == 'train':
            get_train_dataset(self)
        elif split == 'test':
            get_test_dataset(self)
        else:
            get_val_dataset(self)

        self.n_classes = max(self.label) + 1
        image_size = 80
        norm_params = {'mean': [  0.406], 'std': [  0.225]}
        # norm_params = {'mean': [0, 0, 0],  'std': [1, 1, 1]}  #########################################################
        normalize = transforms.Normalize(**norm_params)
        self.default_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),  ##########################################################
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5), ############################改变亮度[0-->2]、对比度[0-->2]、颜色[-0.5-->0.5]
            # transforms.RandomGrayscale(),  ##################################
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        augment = kwargs.get('augment')
        if augment == 'resize':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5), ############################改变亮度[0-->2]、对比度[0-->2]、颜色[-0.5-->0.5]
                # transforms.RandomGrayscale(),  ##################################
                transforms.ToTensor(),
                normalize,
            ])
        elif augment == 'crop':
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size, padding=8),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5), ############################改变亮度[0-->2]、对比度[0-->2]、颜色[-0.5-->0.5]
                # transforms.RandomGrayscale(),  ##################################
                transforms.ToTensor(),
                normalize,
            ])
        elif augment is None:
            self.transform = self.default_transform

        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(3, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(3, 1, 1).type_as(x)
            return x * std + mean

        self.convert_raw = convert_raw

    def __getitem__(self, i):
        return self.transform(self.data[i]), self.label[i]

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, "images_evaluation")) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, "images_background"))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('== Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            file_processed = os.path.join(self.root, self.processed_folder)
            print("== Unzip from " + file_path + " to " + file_processed)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(file_processed)
            zip_ref.close()
        print("Download finished.")


def find_classes(root_dir):
    retour = []
    for (root, dirs, files) in os.walk(root_dir):
        for f in files:
            if (f.endswith("png")):
                r = root.split('/')
                lr = len(r)
                retour.append((f, r[lr - 2] + "/" + r[lr - 1], root))
    # print("== Found %d items " % len(retour))
    return retour


def index_classes(items):
    idx = {}
    for i in items:
        if i[1] not in idx:
            idx[i[1]] = len(idx)
    # print("== Found %d classes" % len(idx))
    return idx

def get_train_dataset(self):
    all_items = find_classes(os.path.join(self.root, self.processed_folder))
    idx_classes = index_classes(all_items)
    data_all = []
    label_all = []
    for index in range(len(all_items)):
        filename = all_items[index][0]
        img_path = str.join('/', [all_items[index][2], filename])
        image = Image.open(img_path)
        image_label = idx_classes[all_items[index][1]]
        data_all.append(image.copy())
        image.close()
        label_all.append(image_label)

    min_label_all = min(label_all)
    label_all = [x - min_label_all for x in label_all]
    for i in range(len(label_all)):
        if 1200 > label_all[i] >= 0:
            self.data.append(data_all[i])
            self.label.append(label_all[i])


def get_test_dataset(self):
    all_items = find_classes(os.path.join(self.root, self.processed_folder))
    idx_classes = index_classes(all_items)
    data_all = []
    label_all = []
    for index in range(len(all_items)):
        filename = all_items[index][0]
        img_path = str.join('/', [all_items[index][2], filename])
        image = Image.open(img_path)
        image_label = idx_classes[all_items[index][1]]
        data_all.append(image.copy())
        image.close()
        label_all.append(image_label)

    min_label_all = min(label_all)
    label_all = [x - min_label_all for x in label_all]
    for i in range(len(label_all)):
        if 1500 > label_all[i] >= 1200:
            self.data.append(data_all[i])
            self.label.append(label_all[i])

    min_label = min(self.label)
    self.label = [x - min_label for x in self.label]

def get_val_dataset(self):
    all_items = find_classes(os.path.join(self.root, self.processed_folder))
    idx_classes = index_classes(all_items)
    data_all = []
    label_all = []
    for index in range(len(all_items)):
        filename = all_items[index][0]
        img_path = str.join('/', [all_items[index][2], filename])
        image = Image.open(img_path)
        image_label = idx_classes[all_items[index][1]]
        data_all.append(image.copy())
        image.close()
        label_all.append(image_label)

    min_label_all = min(label_all)
    label_all = [x - min_label_all for x in label_all]
    for i in range(len(label_all)):
        if 1623 > label_all[i] >= 1500:
            self.data.append(data_all[i])
            self.label.append(label_all[i])
    min_label = min(self.label)
    self.label = [x - min_label for x in self.label]

