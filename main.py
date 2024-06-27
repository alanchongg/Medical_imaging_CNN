from helper import *

def main():
    #hyperameters
    batch_size=45
    num_ep=8
    learning_rate=0.0007
    #augmentations
    aug_bright=(0.2, 0.6)
    aug_contrast=(2, 6)
    aug_saturation=(0.2, 0.6)
    aug_hue=(-0.2, 0.2)

    #helper function starting training process
    helper(batch_size, num_ep, learning_rate, aug_bright, aug_contrast, aug_saturation, aug_hue)

if __name__=='__main__':
    main()