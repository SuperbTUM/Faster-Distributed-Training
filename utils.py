'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import collections

import torch 
import torch.nn as nn
import torch.nn.init as init


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_int = int(seconds)
    seconds_decimal = seconds - seconds_int
    ms = int(seconds_decimal * 1000)

    timestamp = ''
    i = 1
    if days > 0:
        timestamp += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        timestamp += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        timestamp += str(minutes) + 'm'
        i += 1
    if seconds_int > 0 and i <= 2:
        timestamp += str(seconds_int) + 's'
        i += 1
    if ms > 0 and i <= 2:
        timestamp += str(ms) + 'ms'
        i += 1
    if timestamp == '':
        timestamp = '0ms'
    return timestamp


class ProgressBar:
    def __init__(self, bar_length) -> None:
        self.bar_length = bar_length
        self.start_time = 0
        self.last_time = 0

        _, term_width = os.popen('stty size', 'r').read().split()
        self.term_width = int(term_width)
    
    def update(self, curr, total, msg=None):

        # No progress bar 
        if self.bar_length == -1:
            pass 
        # 
        else:
            curr_len = int(self.bar_length * curr / total)
            rest_len = int(self.bar_length - curr_len) - 1
            
            step_time, total_time = self._build_time_record(curr)
            msg = self._build_message(step_time, total_time, msg)

            self._build_progress_bar(curr_len, rest_len, msg)
            self._update_progress_bar(curr, total)
            return 
    
    def _build_progress_bar(self, curr_len, rest_len, msg=None):
        
        sys.stdout.write(' [')

        for _ in range(curr_len):
            sys.stdout.write('=')
        
        sys.stdout.write('>')
        for _ in range(rest_len):
            sys.stdout.write('.')
        
        sys.stdout.write(']')

        if msg:
            sys.stdout.write(msg)
            for _ in range(self.term_width - int(self.bar_length) - len(msg) - 3):
                sys.stdout.write(' ')

        return 

    def _update_progress_bar(self, curr, total):
        # Go back to the center of the bar.
        for _ in range(self.term_width - int(self.bar_length / 2) + 2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d / %d ' % (curr + 1, total))

        if curr < total - 1:
            sys.stdout.write('\r')
        else:
            sys.stdout.write('\n')
        sys.stdout.flush()
        return 

    def _build_time_record(self, curr):
        if curr == 0:
            self.start_time = time.time()
            self.last_time = self.start_time
        
        curr_time = time.time()
        step_time = curr_time - self.last_time
        self.last_time = curr_time 
        total_time = curr_time - self.start_time

        return step_time, total_time

    def _build_message(self, step_time, total_time, msg):
        msg_parts = []
        msg_parts.append(
            '  Step: %s' % format_time(step_time)
        )
        msg_parts.append(
            ' | Total: %s' % format_time(total_time)
        )

        if msg:
            msg_parts.append(' | ' + msg)

        msg = ''.join(msg_parts)

        return msg 


class Metrics:
    def __init__(self) -> None:
        self._double_dict = {}

    def metric(self, key, sub_key, value):
        if key in self._double_dict:
            if sub_key in self._double_dict:
                self._double_dict[key][sub_key].append(value)
            else:
                self._double_dict[key][sub_key] = value 
        else:
            self._double_dict[key] = {}




# def get_mean_and_std(dataset):
#     '''Compute the mean and std value of dataset.'''
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
#     mean = torch.zeros(3)
#     std = torch.zeros(3)
#     print('==> Computing mean and std..')
#     for inputs, targets in dataloader:
#         for i in range(3):
#             mean[i] += inputs[:,i,:,:].mean()
#             std[i] += inputs[:,i,:,:].std()
#     mean.div_(len(dataset))
#     std.div_(len(dataset))
#     return mean, std

# def init_params(net):
#     '''Init layer parameters.'''
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             init.kaiming_normal(m.weight, mode='fan_out')
#             if m.bias:
#                 init.constant(m.bias, 0)
#         elif isinstance(m, nn.BatchNorm2d):
#             init.constant(m.weight, 1)
#             init.constant(m.bias, 0)
#         elif isinstance(m, nn.Linear):
#             init.normal(m.weight, std=1e-3)
#             if m.bias:
#                 init.constant(m.bias, 0)


# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)