# README #

Please Refer to https://github.com/soumith/ganhacks for important tips and tricks.

### What is this repository for? ###

* I am going to implement a simple GAN both in Keras and Tensorflow.
* This simple GAN generates data similar to a Dataset (e.g. MNIST)

### How do I get set up? ###

* First I need to define the discriminator and generator models.
* The generator is fed by a random noise and should output an image of size the desired dataset.
* The discriminator should get two inputs: [fake image, real image]. Then it is trained for a number of steps.
* Next the generator is trained for one iteration, given the updated discriminator. 
  Then we go back to retraining the discriminator, given the updated generator. 
  This goes on and on and on, until the discriminator can not distinguish well between fake and real images.


