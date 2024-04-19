'''
The ImagePool stores at most pool_size of generated images, and after it is full, it 
will either return the newly generated image or one of the stored images(and replace
it with the newly generated image).
This procedure is designed to update the discriminator with a history of generated images
rather than the latest images produced by the generator, which will reduce model oscillation.
'''

import random

class ImagePool():
    def __init__(self, pool_size=50) -> None:
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image):
        # This implementation is based on batch_size=1, so here we only send in one image at a time
        if self.pool_size == 0:
            return image

        if self.num_imgs < self.pool_size:
            self.num_imgs += 1
            self.images.append(image)
            return image
        else:
            # if the pool is full, on 50% cases the pool returns the latest image
            # on the rest 50% cases, the pool returns an earlier image and replaces it with the latest image
            p = random.uniform(0, 1)
            if p > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                return_image = self.images[idx]
                self.images[idx] = image
                return return_image
            else:
                return image