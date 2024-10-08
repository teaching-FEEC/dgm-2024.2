# Metrics

## Fréchet Inception Distance

Original article: [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500)

The _Fréchet distance_ between the Gaussian with mean and covariance ($\mu _1$, $\Sigma _1$) and the Gaussian with mean and covariance ($\mu _2$, $\Sigma _2$) is:

$$
FID = ||\mu _1 - \mu _2||_2^2 + Tr(\Sigma _1 + \Sigma _2 - 2 \sqrt{\Sigma _1 \Sigma _2})
$$

Extracted from https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
```
The FID score uses the inception v3 model. Specifically, the coding layer of the model (the last pooling layer prior to the output classification of images) is used to capture computer-vision-specific features of an input image. These activations are calculated for a collection of real and generated images.

The activations are summarized as a multivariate Gaussian by calculating the mean and covariance of the images. These statistics are then calculated for the activations across the collection of real and generated images.

The distance between these two distributions is then calculated using the Frechet distance, also called the Wasserstein-2 distance.
A lower FID indicates better-quality images; conversely, a higher score indicates a lower-quality image and the relationship may be linear.

The authors of the score show that lower FID scores correlate with better-quality images when systematic distortions were applied such as the addition of random noise and blur.
```