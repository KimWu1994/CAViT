import random
import math
import numpy as np
np.random.seed(2021)


class LoopPadding(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = list(frame_indices)

        while len(out) < self.size:
            for index in out:
                if len(out) >= self.size:
                    break
                out.append(index)

        return out



class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, num=8):
        self.num = num

    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        # size = self.size

        if len(frame_indices) > self.num:
            step = len(frame_indices) // self.num
            end = 0 + self.num*step
            out = frame_indices[0: end: step]

        else:
            out = frame_indices[0:self.num]
            while len(out) < self.num:
                for index in out:
                    if len(out) >= self.num:
                        break
                    out.append(index)
            out = sorted(out)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, num=8):
        self.num = num

    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        # size = self.size

        if len(frame_indices) > self.num:
            step = len(frame_indices) // self.num
            start = np.random.randint(step)
            end = start + self.num * step
            out = frame_indices[start: end: step]

        else:
            out = frame_indices[0:self.num]
            while len(out) < self.num:
                for index in out:
                    if len(out) >= self.num:
                        break
                    out.append(index)
            out = sorted(out)
        return out


class TemporalRandomContinueCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, num=8):
        self.num = num

    def __call__(self, frame_indices):
        frame_indices = list(frame_indices)
        # size = self.size

        if len(frame_indices) > self.num:

            start = np.random.randint(len(frame_indices) - self.num)
            # end = start + self.num * step
            out = frame_indices[start: start+self.num]

        else:
            out = frame_indices[0:self.num]
            while len(out) < self.num:
                for index in out:
                    if len(out) >= self.num:
                        break
                    out.append(index)
            out = sorted(out)
        return out

