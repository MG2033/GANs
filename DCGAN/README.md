# DCGAN
DCGANs were introduced in the paper called [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
](https://arxiv.org/abs/1511.06434). They are the true beginning of the new era of GANs after the first paper ([Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)) published by Ian Goodfellow et al.

## Result
The implementation of this network was tested on celebA dataset with a ratio between the discriminator runs to the generator runs of 5:1. Here is a sample output at epoch #100 to prove that it's working.
<div align="center">
<img src="https://github.com/MG2033/GANs/blob/master/figures/samples_epoch_99.png"><br><br>
</div>

## File Structure
- main.py  : It's the main program from which training or testing starts.
- model.py : The implementation of the network itself is in this file.
- train.py : Loading the data, constructing the loss, performing training procedure, and performing testing procedure are implemented here.

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

