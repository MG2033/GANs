# GANs
An implementation of several generative adversarial networks introduced in PyTorch. 

This implementation was made to be an example of a common deep learning software architecture. It's simple and designed to be very modular. All of the components needed for training and visualization are added.

## Released Models
[DCGAN](https://github.com/MG2033/GANs/DCGAN/README.md)

## Usage
This project uses Python 3.5.3 and PyTorch 0.3.

### Main Dependencies
 ```
 pytorch 0.4
 numpy 1.13.1
 tqdm 4.15.0
 easydict 1.7
 matplotlib 2.0.2
 tensorboardX 1.0
 ```

### Run
```
python main.py --config configs/<your-config-json-file>.json
```

#### Tensorboard Visualization
Tensorboard is integrated with the project using `tensorboardX` library which proved to be very useful as there is no official visualization library in pytorch.

You can start it using:
```bash
tensorboard --logdir experiments/<config-name>/summaries
```

## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

