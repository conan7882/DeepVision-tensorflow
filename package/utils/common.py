import scipy.misc
import numpy as np

__all__ = ['save_images', 'get_file_list']

def save_images(images, im_size, save_path):
    
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """

    # normalization of tanh output
    img = (images + 1.0) / 2.0
    # img = images
    h, w = img.shape[1], img.shape[2]

    merge_img = np.zeros((h * im_size[0], w * im_size[1], 3))

    for idx, image in enumerate(images):
        i = idx % im_size[1]
        j = idx // im_size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image
    
    return scipy.misc.imsave(save_path, merge_img)

def get_file_list(file_dir, file_ext):
    assert file_ext in ['.mat', '.png', '.jpg']
    return file_list = np.array([os.path.join(file_dir, file) 
        for file in os.listdir(file_dir) 
        if file.endswith(file_ext)])
