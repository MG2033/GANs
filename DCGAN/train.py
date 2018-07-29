from tensorboardX import SummaryWriter
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import AverageMeter


class DCGANTrainer:
    def __init__(self, generator, discriminator, data_loader, device, summary_dir, checkpoint_dir, output_dir, config):
        # Initialize the configurations
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

        # Initialize the generator, discriminator, and the data loader
        self.generator = generator.to(self.device)
        self.discrimintator = discriminator.to(self.device)
        self.data_loader = data_loader

        # Initialize the epoch counter
        self.epoch = 0
        self.iteration = 0

        self.fixed_z_input = torch.randn(self.config.fixed_Z_batch_size, self.config.Z_dim, 1, 1, device=self.device,
                                         dtype=torch.float32)

        # Create the loss and the optimizer
        if self.config.mode == "train":
            self.loss = None
            self.generator_optimizer = None
            self.discrimintator_optimizer = None
            self.create_loss()

        # Load a previous checkpoint
        self.load(self.config.checkpoint_file_name)

        # Create the tensorboardX summary writer
        self.summary_writer = SummaryWriter(summary_dir)

    def save(self, filename="checkpoint.pth.tar", is_best=False):
        state = {
            'epoch': self.epoch + 1,
            'iteration': self.iteration,
            'fixed_z_input': self.fixed_z_input,
            'discriminator_state_dict': self.discrimintator.state_dict(),
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_optimizer': self.discrimintator_optimizer.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.checkpoint_dir + filename)
        # If it is the best, copy it to another file.
        if is_best:
            shutil.copyfile(self.checkpoint_dir + filename,
                            self.checkpoint_dir + 'model_best.pth.tar')

    def load(self, filename="checkpoint.pth.tar"):
        filename = self.checkpoint_dir + filename
        try:
            print("Loading checkpoint from '{}'".format(filename))
            state = torch.load(filename)

            self.epoch = state['epoch']
            self.iteration = state['iteration']
            self.fixed_z_input = state['fixed_z_input']
            self.discrimintator.load_state_dict(state['discriminator_state_dict'])
            self.generator.load_state_dict(state['generator_state_dict'])
            self.discrimintator_optimizer.load_state_dict(state['discriminator_optimizer'])
            self.generator_optimizer.load_state_dict(state['generator_optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}) - (iteration {})\n"
                  .format(self.checkpoint_dir, state['epoch'], state['iteration']))
        except OSError as _:
            print("No checkpoint exists at '{}'. Skipping...".format(self.checkpoint_dir))
            print("First time to train...\n")

    def train(self):
        for cur_epoch in range(self.epoch, self.config.num_epochs):
            # Increment current epoch
            self.epoch = cur_epoch
            # Test using a predetermined Z vector from the latent space (before)
            self.test_fixed_Z()
            # Train for one epoch
            self.train_one_epoch(cur_epoch)
            # Save the model checkpoint
            self.save(self.config.checkpoint_file_name)
            # Test using a predetermined Z vector from the latent space (after)
            self.test_fixed_Z()

        self.summary_writer.close()
        self.data_loader.make_gif(self.epoch, self.output_dir)

    def train_one_epoch(self, cur_epoch):
        # Initialize tqdm using the data loader
        tqdm_batch = tqdm(self.data_loader.train_loader,
                          desc="Epoch-" + str(cur_epoch) + "-")

        # Put the generator and the discriminator in training mode (for batch normalization and dropout)
        self.generator.train()
        self.discrimintator.train()

        # Create the real and the fake labels
        y_real = torch.ones(self.config.batch_size, device=self.device) * self.config.real_label
        y_fake = torch.ones(self.config.batch_size, device=self.device) * self.config.fake_label

        # Create the meters that log the losses and the scores
        generator_loss_meter, discriminator_loss_meter = AverageMeter(), AverageMeter()
        score_real_meter, score_fake_before_meter, score_fake_after_meter = AverageMeter(), AverageMeter(), AverageMeter()

        for real_batch, _ in tqdm_batch:
            if real_batch.shape[0] < self.config.batch_size:
                break
            # Create the random noise vector from the latent space Z
            z_input = torch.randn(self.config.batch_size, self.config.Z_dim, 1, 1, device=self.device,
                                  dtype=torch.float32)

            # TODO: Training the discriminator using a ratio N:1 to the generator
            # Train the discriminator using real data
            real_batch = real_batch.to(self.device)
            out = self.discrimintator(real_batch)
            discriminator_loss_real = self.loss(out, y_real)
            score_real = out.data.mean()

            # Train the discriminator using fake data
            gen_out = self.generator(z_input)
            out = self.discrimintator(gen_out.detach())
            discriminator_loss_fake = self.loss(out, y_fake)
            score_fake_before = out.data.mean()

            # Updating the discriminator
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            self.discrimintator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discrimintator_optimizer.step()

            # Training the generator to fool the discriminator
            out = self.discrimintator(gen_out)
            score_fake_after = out.data.mean()

            # Updating the generator
            generator_loss = self.loss(out, y_real)
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            # Update all metrics
            discriminator_loss_meter.update(discriminator_loss.item())
            generator_loss_meter.update(generator_loss.item())
            score_fake_before_meter.update(score_fake_before.item())
            score_fake_after_meter.update(score_fake_after.item())
            score_real_meter.update(score_real.item())

            self.iteration += 1

        tqdm_batch.close()

        # Add summaries to tensorboard
        self.summary_writer.add_scalar('train/Discriminator_Loss', discriminator_loss_meter.avg, self.iteration)
        self.summary_writer.add_scalar('train/Generator_Loss', generator_loss_meter.avg, self.iteration)
        self.summary_writer.add_scalar('train/Score_Real', score_real_meter.avg, self.iteration)
        self.summary_writer.add_scalar('train/Score_Fake_Before', score_fake_before_meter.avg, self.iteration)
        self.summary_writer.add_scalar('train/Score_Fake_After', score_fake_after_meter.avg, self.iteration)

        # Print Summaries
        epoch_log_str = "Epoch {}: Iteration {}: G_Loss={}, D_Loss={}, Score_Real={}, Score_Fake={}/{}".format(
            cur_epoch, self.iteration, generator_loss_meter.avg, discriminator_loss_meter.avg, score_real_meter.avg,
            score_fake_before_meter.avg, score_fake_after_meter.avg)
        print(epoch_log_str)
        self.summary_writer.add_text("train/Epoch_Log", epoch_log_str, self.iteration)

    def test_fixed_Z(self):
        # Put the generator in evaluation mode
        self.generator.eval()
        # Perform Inference in the generator
        out = self.generator(self.fixed_z_input)
        # Plot the results and log them as well
        self.data_loader.plot_samples_per_epoch(out.data, self.epoch, self.output_dir)
        self.summary_writer.add_image('train/fixed_z_img_epoch', out, self.iteration)

    def create_loss(self):
        # Binary cross entropy loss is used
        self.loss = nn.BCELoss().to(self.device)
        # Adam optimizer is used for the generator and the discriminator
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=self.config.learning_rate,
                                              betas=self.config.adam_betas)
        self.discrimintator_optimizer = optim.Adam(self.discrimintator.parameters(), lr=self.config.learning_rate,
                                                   betas=self.config.adam_betas)
