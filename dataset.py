from torchvision import transforms
import torchvision.datasets as dset
import cv2
from tqdm import tqdm
import numpy as np
import torch
import random
import copy
from copy import deepcopy
from PIL import Image

dataset_names = ['liberty', 'notredame', 'yosemite']
mean_image = 0.443728476019
std_image = 0.20197947209
rotate_degree = 10


class TripletPhotoTour(dset.PhotoTour):
    """
    From the PhotoTour Dataset it generates triplet samples
    note: a triplet is composed by a pair of matching images and one of
    different class.
    """

    def __init__(self, train=True, transform=None, batch_size=None, load_random_triplets=False, args=None, *arg, **kw):
        super(TripletPhotoTour, self).__init__(*arg, **kw)
        self.transform = transform
        self.out_triplets = load_random_triplets
        self.train = train
        self.n_triplets = args.n_triplets
        self.batch_size = batch_size
        self.args = args

        if self.train:
            print('Generating {} triplets'.format(self.n_triplets))
            self.triplets = self.generate_triplets(self.labels, self.n_triplets)

    def generate_triplets(self, labels, num_triplets):
        def create_indices(_labels):
            inds = dict()
            for idx, ind in enumerate(_labels):
                if ind not in inds:
                    inds[ind] = []
                inds[ind].append(idx)
            return inds

        triplets = []
        indices = create_indices(labels.numpy())
        unique_labels = np.unique(labels.numpy())
        n_classes = unique_labels.shape[0]
        # add only unique indices in batch
        already_idxs = set()

        for x in tqdm(range(num_triplets)):
            if len(already_idxs) >= self.args.batch_size:
                already_idxs = set()
            c1 = np.random.randint(0, n_classes)
            while c1 in already_idxs:
                c1 = np.random.randint(0, n_classes)
            already_idxs.add(c1)
            c2 = np.random.randint(0, n_classes)
            while c1 == c2:
                c2 = np.random.randint(0, n_classes)
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]))
                n2 = np.random.randint(0, len(indices[c1]))
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]))
            n3 = np.random.randint(0, len(indices[c2]))
            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3]])
        return torch.LongTensor(np.array(triplets))

    def __getitem__(self, index):
        def transform_img(img):
            if self.transform is not None:
                img = self.transform(img.numpy())
            return img

        if not self.train:
            m = self.matches[index]
            img1 = transform_img(self.data[m[0]])
            img2 = transform_img(self.data[m[1]])
            return img1, img2, m[2]

        t = self.triplets[index]
        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a)
        img_p = transform_img(p)
        img_n = None
        if self.out_triplets:
            img_n = transform_img(n)
        # transform images if required
        if self.args.fliprot:
            do_flip = random.random() > 0.5
            do_rot = random.random() > 0.5
            if do_rot:
                img_a = img_a.permute(0, 2, 1)
                img_p = img_p.permute(0, 2, 1)
                if self.out_triplets:
                    img_n = img_n.permute(0, 2, 1)
            if do_flip:
                img_a = torch.from_numpy(deepcopy(img_a.numpy()[:, :, ::-1]))
                img_p = torch.from_numpy(deepcopy(img_p.numpy()[:, :, ::-1]))
                if self.out_triplets:
                    img_n = torch.from_numpy(deepcopy(img_n.numpy()[:, :, ::-1]))
        if self.out_triplets:
            return (img_a, img_p, img_n)
        else:
            return (img_a, img_p)

    def __len__(self):
        if self.train:
            return self.triplets.size(0)
        else:
            return self.matches.size(0)


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR)


def create_loaders(args, load_random_triplets=False):
    test_dataset_names = copy.copy(dataset_names)
    test_dataset_names.remove(args.training_set)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    cv2_scale = lambda x: cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
    np_reshape = lambda x: np.reshape(x, (32, 32, 1))
    np_reshape64 = lambda x: np.reshape(x, (64, 64, 1))
    cv2_resize = lambda x: cv2.resize(np.array(x),dsize=(32, 32), interpolation=cv2.INTER_LINEAR)

    transform_test = transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        transforms.Normalize((mean_image,), (std_image,))])
    transform_train = transforms.Compose([
        transforms.Lambda(np_reshape64),
        transforms.ToPILImage(),
        RandomRotate(rotate_degree),
        transforms.RandomResizedCrop(32, scale=(0.9, 1.0), ratio=(0.9, 1.1)),
        transforms.Lambda(cv2_resize),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        transforms.Normalize((mean_image,), (std_image,))])
    transform = transforms.Compose([
        transforms.Lambda(cv2_scale),
        transforms.Lambda(np_reshape),
        transforms.ToTensor(),
        transforms.Normalize((mean_image,), (std_image,))])
    if not args.augmentation:
        transform_train = transform
        transform_test = transform
    train_loader = torch.utils.data.DataLoader(
        TripletPhotoTour(train=True,
                         load_random_triplets=load_random_triplets,
                         batch_size=args.batch_size,
                         root=args.dataroot,
                         name=args.training_set,
                         download=True,
                         args=args,
                         transform=transform_train),
        batch_size=args.batch_size,
        shuffle=False, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
                         TripletPhotoTour(train=False,
                                          batch_size=args.batch_size,
                                          root=args.dataroot,
                                          name=name,
                                          args=args,
                                          download=True,
                                          transform=transform_test),
                         batch_size=args.batch_size,
                         shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, test_loaders
