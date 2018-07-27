import imageio
import torchvision.utils as v_utils


class BaseDataLoader:
    def plot_samples_per_epoch(self, fake_batch, epoch, output_dir):
        v_utils.save_image(fake_batch,
                           '{}samples_epoch_{:d}.png'.format(output_dir, epoch),
                           nrow=5,
                           padding=2,
                           normalize=True)

    def make_gif(self, epochs, output_dir):
        gen_image_plots = []
        for epoch in range(epochs + 1):
            img_epoch = '{}samples_epoch_{:d}.png'.format(output_dir, epoch)
            gen_image_plots.append(imageio.imread(img_epoch))

        imageio.mimsave(output_dir + 'animation_epochs_{:d}.gif'.format(epochs), gen_image_plots, fps=2)
