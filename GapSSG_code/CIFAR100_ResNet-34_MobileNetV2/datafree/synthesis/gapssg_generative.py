import datafree
from typing import Generator
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import random

from .base import BaseSynthesis
from datafree.hooks import DeepInversionHook, InstanceMeanHook
from datafree.criterions import jsdiv, get_image_prior_losses, kldiv
from datafree.utils import ImagePool, DataIter, clip_images
import collections
from torchvision import transforms
from kornia import augmentation




class MLPHead(nn.Module):
    def __init__(self, dim_in, dim_feat, dim_h=None):
        super(MLPHead, self).__init__()
        if dim_h is None:
            dim_h = dim_in

        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_h),
            nn.ReLU(inplace=True),
            nn.Linear(dim_h, dim_feat),
        )

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, dim=1, p=2)

class MultiTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform
    
    def __call__(self, x):
        return [t(x) for t in self.transform]

    def __repr__(self):
        return str( self.transform )




class MemoryBank(object):
    def __init__(self, device, max_size=4096, dim_feat=512):
        self.device = device
        self.data = torch.randn( max_size, dim_feat ).to(device)
        self._ptr = 0
        self.n_updates = 0

        self.max_size = max_size
        self.dim_feat = dim_feat

    def add(self, feat):
        feat = feat.to(self.device)
        n, c = feat.shape
        assert self.dim_feat==c and self.max_size % n==0, "%d, %d"%(self.dim_feat, c, self.max_size, n)
        self.data[self._ptr:self._ptr+n] = feat.detach()
        self._ptr = (self._ptr+n) % (self.max_size)
        self.n_updates+=n

    def get_data(self, k=None, index=None):
        if k is None:
            k = self.max_size

        if self.n_updates>self.max_size:
            if index is None:
                index = random.sample(list(range(self.max_size)), k=k)
            return self.data[index], index
        else:
            #return self.data[:self._ptr]
            if index is None:
                index = random.sample(list(range(self._ptr)), k=min(k, self._ptr))
            return self.data[index], index

def reset_model(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class Synthesizer(BaseSynthesis):
    def __init__(self, teacher, student, generator, nz, num_classes, img_size, 
                 feature_layers=None, bank_size=40960, n_neg=4096, head_dim=128, init_dataset=None,
                 iterations=100, lr_g=0.1, progressive_scale=False,
                 synthesis_batch_size=128, sample_batch_size=128, 
                 save_dir='run/gapssg', transform=None,
                 autocast=None, use_fp16=False, 
                 normalizer=None, device='cpu', distributed=False):
        super(Synthesizer, self).__init__(teacher, student)
        self.save_dir = save_dir
        self.img_size = img_size 
        self.iterations = iterations
        self.lr_g = lr_g
        self.progressive_scale = progressive_scale
        self.nz = nz
        self.n_neg = n_neg
        self.num_classes = num_classes
        self.distributed = distributed
        self.synthesis_batch_size = synthesis_batch_size
        self.sample_batch_size = sample_batch_size
        self.bank_size = bank_size
        self.init_dataset = init_dataset

        self.use_fp16 = use_fp16
        self.autocast = autocast # for FP16
        self.normalizer = normalizer
        self.data_pool = ImagePool(root=self.save_dir)
        self.transform = transform
        self.data_iter = None

        self.cmi_hooks = []
        if feature_layers is not None:
            for layer in feature_layers:
                self.cmi_hooks.append( InstanceMeanHook(layer) )
        else:
            for m in teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    self.cmi_hooks.append( InstanceMeanHook(m) )

        with torch.no_grad():
            teacher.eval()
            fake_inputs = torch.randn(size=(1, *img_size), device=device)
            _ = teacher(fake_inputs)
            cmi_feature = torch.cat([ h.instance_mean for h in self.cmi_hooks ], dim=1)
            print("dims: %d"%(cmi_feature.shape[1]))
            del fake_inputs
        
        self.generator = generator.to(device).train()
        # local and global bank
        self.mem_bank = MemoryBank('cpu', max_size=self.bank_size, dim_feat=2*cmi_feature.shape[1]) # local + global
        
        self.head = MLPHead(cmi_feature.shape[1], head_dim).to(device).train()
        self.optimizer_head = torch.optim.Adam(self.head.parameters(), lr=self.lr_g)

        self.device = device
        self.hooks = []
        for m in teacher.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.hooks.append( DeepInversionHook(m) )

        self.aug = MultiTransform([
            # global view
            transforms.Compose([ 
                augmentation.RandomCrop(size=[self.img_size[-2], self.img_size[-1]], padding=4),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
            # local view
            transforms.Compose([
                augmentation.RandomResizedCrop(size=[self.img_size[-2], self.img_size[-1]], scale=[0.25, 1.0]),
                augmentation.RandomHorizontalFlip(),
                normalizer,
            ]),
        ])



    def synthesize(self, targets=None):
        self.student.eval()
        self.teacher.eval()
        best_cost = 1e6
        
        #inputs = torch.randn( size=(self.synthesis_batch_size, *self.img_size), device=self.device ).requires_grad_()
        best_inputs = None
        z = torch.randn(size=(self.synthesis_batch_size, self.nz), device=self.device).requires_grad_() 
        if targets is None:
            targets = torch.randint(low=0, high=self.num_classes, size=(self.synthesis_batch_size,))
            targets = targets.sort()[0] # sort for better visualization
        targets = targets.to(self.device)

        reset_model(self.generator)
        optimizer = torch.optim.Adam([{'params': self.generator.parameters()}, {'params': [z]}], self.lr_g, betas=[0.5, 0.999])
        for it in range(self.iterations):
            inputs = self.generator(z)
            global_view, local_view = self.aug(inputs) # crop and normalize

            #############################################
            # Generation Loss
            #############################################
            t_out = self.teacher(global_view)
            loss_bn = sum([h.r_feature for h in self.hooks])
            s_out = self.student(global_view)
            loss_ds = F.cross_entropy( (t_out-s_out)/2, targets )
            loss_as = F.cross_entropy( t_out+s_out, targets )
            p_s = F.softmax(s_out.detach(), dim=1)
            p_t = F.softmax(t_out.detach(), dim=1)
            pst_max, indx = torch.abs(p_s - p_t).max(1)
            scalar_grad = torch.mean( pst_max )
            scalar_grad = torch.exp( - scalar_grad )
            loss_inv = 1 * loss_bn + 0.5 * (1 - scalar_grad) * (loss_ds + 0.2 * loss_as)
            loss =  loss_inv
            
            with torch.no_grad():
                if best_cost > loss.item() or best_inputs is None:
                    best_cost = loss.item()
                    best_inputs = inputs.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        self.student.train()
        # save best inputs and reset data iter
        self.data_pool.add( best_inputs )

        dst = self.data_pool.get_dataset(transform=self.transform)
        if self.init_dataset is not None:
            init_dst = datafree.utils.UnlabeledImageDataset(self.init_dataset, transform=self.transform)
            dst = torch.utils.data.ConcatDataset([dst, init_dst])
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dst)
        else:
            train_sampler = None
        loader = torch.utils.data.DataLoader(
            dst, batch_size=self.sample_batch_size, shuffle=(train_sampler is None),
            num_workers=4, pin_memory=True, sampler=train_sampler)
        self.data_iter = DataIter(loader)
        return {"synthetic": best_inputs}
        
    def sample(self):
        return self.data_iter.next()