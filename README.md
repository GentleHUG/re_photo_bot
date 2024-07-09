# re_photo_bot
telegram bot for redacting photos

The repository can be divided into 2 parts: model training and telegram bot.

# 1.  Model training

To prepare the datasets, we load them into Data Loader separately, which can then iterate through the datasets as needed. Because the training dataset contains both the Monet paintings and photos, we pass both data loaders into CombinedLoader for training.
We check that the data module defined is working as intended by visualizing samples of the images below
Building GAN Architecture¶
U-Net Generator

A common architecture for the CycleGAN generator is the U-Net. U-Net is a network which consists of a sequence of downsampling blocks followed by a sequence of upsampling blocks, giving it the U-shaped architecture. In the upsampling path, we concatenate the outputs of the upsampling blocks and the outputs of the downsampling blocks symmetrically. This can be seen as a kind of skip connection, facilitating information flow in deep networks and reducing the impact of vanishing gradients.

ResNet generator

Similar to the U-Net architecture, the ResNet generator consists of the downsampling path and upsampling path. The difference is that the ResNet generator does not have the long skip connections from the concatenations of outputs. Instead, the ResNet generator uses residual blocks between the two paths. These residual blocks have short skip connections where the original input is added to the output.

Downsampling blocks

The downsampling blocks use convolution layers to increase the number of feature maps while reducing the dimensions of the 2D image.
Upsampling blocks

On the other hand, the upsampling blocks contain transposed convolution layers, which combine the learned features to output an image with the original size.
Residual blocks

As described above, the residual blocks have convolution layers where the original input is added to the output.
Building the generator

With the building blocks defined, we can now build our CycleGAN generator. For reference, the output size of each block is commented below.
Patch Gan Generator

Unlike conventional networks that output a single probability of the input image being real or fake, CycleGAN uses the PatchGAN discriminator that outputs a matrix of values. Intuitively, each value of the output matrix checks the corresponding portion of the input image. Values closer to 1 indicate real classification and values closer to 0 indicate fake classification.

Building the discriminator

In general, the PatchGAN discriminator consists of a sequence of convolution layers, which can be built using the downsampling blocks defined earlier.
CycleGan

With the generator and discriminator architectures defined, we can now build CycleGAN, which consists of two generators and two discriminators:

Generator for photo-to-Monet translation (gen_PM).
Generator for Monet-to-photo translation (gen_MP).
Discriminator for Monet paintings (disc_M).
Discriminator for photos (disc_P). We use the Adam optimizer for model training. To optimize the parameters, we need to define the loss functions:

Discriminator loss:- For real images fed into the discriminator, the output matrix is compared against a matrix of 1s using the mean squared error. For fake images, the output matrix is compared against a matrix of 0s. This suggests that to minimize loss, the perfect discriminator outputs a matrix of 1s for real images and a matrix of 0s for fake images.

Generator loss:- This is composed of three different loss functions below.
Adversarial loss :-Fake images are fed into the discriminator and the output matrix is compared against a matrix of 1s using the mean squared error. To minimize loss, the generator needs to 'fool' the discriminator into thinking that the fake images are real and output a matrix of 1s.
Identity loss:- When a Monet painting is fed into the photo-to-Monet generator, we should get back the same Monet painting because nothing needs to be transformed. The same applies for photos fed into the Monet-to-photo generator. To encourage identity mapping, the difference in pixel values between the input image and generated image is measured using the l1 loss.
Cycle loss When a Monet painting is fed into the Monet-to-photo generator, and the generated image is fed back into the photo-to-Monet generator, it should transform back into the original Monet painting. The same applies for photos passed to the two generators to get back the original photos. To preserve information throughout this cycle, the l1 loss is used to measure the difference between the original image and the reconstructed image.
Building the CycleGAN model
Computing the predictions can be done by running the predict method to generate the Monet-style images given the input photos.
