# -*- coding: utf-8 -*- 
import numpy as np
import torch
import random
from IPython import embed

class AddGaussianNoise(object):
    """
    Args:
        miu （float)
        sigma (float)
        p(float): probability of implement this transform
    """

    def __init__(self, miu=0, sigma=0.1, p=1.0):
        self.miu = miu
        self.sigma = sigma
        self.p = p

    def __call__(self, input, target, hierarchy):
        """
        Args:
            input： 3d dense Image
        Returns:
            3d dense Image
        """

        if random.uniform(0, 1) < self.p:
            h, w, c = input.shape[-3:]
            gaussian_noise = noise = np.random.randn(h, w, c) *self.sigma + self.miu 
            input += gaussian_noise
 
        return input, target, hierarchy
        
class AddPepperNoise(object):
    """
    Args:
        snr （float）: Signal Noise Rate
        p(float): probability of implement this transform
    """

    def __init__(self, snr, p=0.95):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, input, target, hierarchy):
        """
        Args:
            input： 3d dense Image
        Returns:
            3d dense Image
        """

        if random.uniform(0, 1) < self.p:
            # input_ = np.array(input).copy()
            h, w, c = input.shape[-3:]
            signal_pct = self.snr
            noise_pct = (1 - self.snr)
            gaussian_noise = noise = np.random.randn(h, w, c).clip(-1, 1) * 2 #[-3, 3]
            mask = np.random.choice((0, 1, 2), size=(h, w, c), p=[signal_pct, noise_pct/2., noise_pct/2.])
            # mask = np.repeat(mask, c, axis=2)
            input[mask == 1] = -float('inf')  
            # input[mask == 2] = -float('inf')
            input[mask == 2] = gaussian_noise[mask == 2]    

        return input, target, hierarchy

class AddRandomFlip(object):
    """
    Args:
        p0: HorizontalFlip
        p1: VerticalFlip
        p2: Exchange X & Y
    """
    def __init__(self, p0=0.5, p1=0.5, p2=0.5):
        assert isinstance(p0, float) and (isinstance(p0, float)) and (isinstance(p2, float))
        self.p0 = p0
        self.p1 = p1
        self.p2 = p2
        np.random.seed(7)

    def __call__(self, input, target, hierarchy):
        """
        Args:
            input： 3d dense Image (numpy array)
        Returns:
            3d dense Image
        """
        if np.random.rand(1) < self.p0:
            input = input[:, ::-1, :]
            if target is not None:
                target = target[:, ::-1, :]
            if hierarchy is not None:
                for h in hierarchy:
                    h = h[:, ::-1, :]

        
        if np.random.rand(1) < self.p1:
            input = input[:, :, ::-1]
            if target is not None:
                target = target[:, :, ::-1]
            if hierarchy is not None:
                for h in hierarchy:
                    h = h[:, :, ::-1]

        if np.random.rand(1) < self.p2:
            input = input.swapaxes(-3, -2)
            if target is not None:
                target = target.swapaxes(-3, -2)
            if hierarchy is not None:
                for h in hierarchy:
                    h = h.swapaxes(-3, -2)

        input = input.copy()
        if target is not None:
            target = target.copy()
        if hierarchy is not None:
            for h in hierarchy:
                h = h.copy()

        return input, target, hierarchy

class RandomliftFloor(object):
    """
    Args:
        p: probability of implement change
    """
    def __init__(self, p=0.8):
        assert isinstance(p, float)
        self.p = p
        self.default_value = -float('inf')

    def __call__(self, input, target, hierarchy, lift_step):
        """
        Args:
            input： 3d dense Image (numpy array)
        Returns:
            3d dense Image
        """
            
        if np.random.rand(1) < self.p:
            # raise
            # print("lift step: ", lift_step)
            if lift_step > 0:
                
                input[:, :, lift_step:] = input[:, :, :-lift_step]
                input[:, :, :lift_step] = self.default_value

                if target is not None:
                    target[:, :, lift_step:] = target[:, :, :-lift_step]
                    target[:, :, :lift_step] = self.default_value   

                lift_step /= (2 ** len(hierarchy))
                if hierarchy is not None:
                    for h in hierarchy:
                        h[:, :, lift_step:] = h[:, :, :-lift_step]
                        h[:, :, :-lift_step] = self.default_value 
                        lift_step *= 2

            elif lift_step < 0:
                lift_step = -lift_step
                # sink
                input[:, :, :-lift_step] = input[:, :, lift_step:]
                input[:, :, -lift_step:] = self.default_value

                if target is not None:
                    target[:, :, :-lift_step] = target[:, :, lift_step:]
                    target[:, :, -lift_step:] = self.default_value   

                lift_step /= (2 ** len(hierarchy))
                if hierarchy is not None:
                    for h in hierarchy:
                        h[:, :, :-lift_step] = h[:, :, lift_step:]
                        h[:, :, -lift_step:] = self.default_value                          
                        lift_step *= 2

        return [input, target, hierarchy]

class MyTransforms():

    def __init__(self, transform_list, random_lift=None, max_lift=2):
        self.trans = transform_list

        self.lift = random_lift
        self.max_lift = max_lift # max_lift / (2 ^ num_hierarchy) should be int ## acturally max_lift * 2 ^ num_hierarchy

    def __call__(self, input, target, hierarchy):
        for t in self.trans:
            # print("before: ", input.shape)
            input, target, hierarchy = t(input, target, hierarchy)

        if self.lift:
            lift_step = np.random.randint(-0, self.max_lift) * 4
            input, target, hierarchy = self.lift(input, target, hierarchy, lift_step)

        return input, target, hierarchy    