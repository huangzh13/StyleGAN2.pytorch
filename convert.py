"""
-------------------------------------------------
   File Name:    convert.py.py
   Author:       Zhonghao Huang
   Date:         2019/12/13
   Description:  
-------------------------------------------------
"""

import torch

if __name__ == '__main__':
    input_file = './weights/stylegan2-ffhq-config-f-torch.pt'

    state_G, state_D, state_Gs, dlatent_avg = torch.load(input_file)

    print('Done.')
