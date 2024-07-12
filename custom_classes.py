# custom_classes.py

# import deepspeed as ds
import numpy as np
import pytorch_lightning as L
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import read_image


class CustomTransform(object):
    def __init__(self, load_dim=286, target_dim=256):
        self.transform_train = T.Compose([

            T.Resize((load_dim, load_dim), antialias=True),
            T.RandomCrop((target_dim, target_dim)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        self.transform = T.Resize((target_dim, target_dim), antialias=True)

    def __call__(self, img, stage):
        if stage == "fit":
            img = self.transform_train(img)
        else:
            # print(img.shape)
            img = self.transform(img)
            # print(img.shape)

        return img * 2 - 1

class CustomDataset(Dataset):
    def __init__(self, filenames, transform, stage):
        self.filenames = filenames
        self.transform = transform
        self.stage = stage

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        img = read_image(img_name) / 255.0
        return self.transform(img, stage=self.stage)


   
class Downsampling(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            norm=True,
            lrelu=True,
        ):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding, bias=not norm),
            )
            if norm:
                self.block.append(nn.InstanceNorm2d(out_channels, affine=True))
            if lrelu is not None:
                self.block.append(nn.LeakyReLU(0.2, True) if lrelu else nn.ReLU(True))
            
        def forward(self, x):
            return self.block(x)

class Upsampling(nn.Module):
        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            output_padding=0,
            dropout=False,
        ):
            super().__init__()
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride, 
                                padding=padding, output_padding=output_padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
            )
            if dropout:
                self.block.append(nn.Dropout(0.5))
            self.block.append(nn.ReLU(True))
            
        def forward(self, x):
            return self.block(x)
class ResBlock(nn.Module):
        def __init__(self, in_channels, kernel_size=3, padding=1):
            super().__init__()
            self.block = nn.Sequential(
                nn.ReflectionPad2d(padding),
                Downsampling(in_channels, in_channels,
                            kernel_size=kernel_size, stride=1, padding=0, lrelu=False),
                nn.ReflectionPad2d(padding),
                Downsampling(in_channels, in_channels,
                            kernel_size=kernel_size, stride=1, padding=0, lrelu=None),
            )
            
        def forward(self, x):
            return x + self.block(x)

class UNetGenerator(nn.Module):
        def __init__(self, hid_channels, in_channels, out_channels):
            super().__init__()
            self.downsampling_path = nn.Sequential(
                Downsampling(in_channels, hid_channels, norm=False), # 64x128x128
                Downsampling(hid_channels, hid_channels*2), # 128x64x64
                Downsampling(hid_channels*2, hid_channels*4), # 256x32x32
                Downsampling(hid_channels*4, hid_channels*8), # 512x16x16
                Downsampling(hid_channels*8, hid_channels*8), # 512x8x8
                Downsampling(hid_channels*8, hid_channels*8), # 512x4x4
                Downsampling(hid_channels*8, hid_channels*8), # 512x2x2
                Downsampling(hid_channels*8, hid_channels*8, norm=False), # 512x1x1, instance norm does not work on 1x1
            )
            self.upsampling_path = nn.Sequential(
                Upsampling(hid_channels*8, hid_channels*8, dropout=True), # (512+512)x2x2
                Upsampling(hid_channels*16, hid_channels*8, dropout=True), # (512+512)x4x4
                Upsampling(hid_channels*16, hid_channels*8, dropout=True), # (512+512)x8x8
                Upsampling(hid_channels*16, hid_channels*8), # (512+512)x16x16
                Upsampling(hid_channels*16, hid_channels*4), # (256+256)x32x32
                Upsampling(hid_channels*8, hid_channels*2), # (128+128)x64x64
                Upsampling(hid_channels*4, hid_channels), # (64+64)x128x128
            )
            self.feature_block = nn.Sequential(
                nn.ConvTranspose2d(hid_channels*2, out_channels,
                                kernel_size=4, stride=2, padding=1), # 3x256x256
                nn.Tanh(),
            )
            
        def forward(self, x):
            skips = []
            for down in self.downsampling_path:
                x = down(x)
                skips.append(x)
            skips = reversed(skips[:-1])

            for up, skip in zip(self.upsampling_path, skips):
                x = up(x)
                x = torch.cat([x, skip], dim=1)
            return self.feature_block(x)
        
class ResNetGenerator(nn.Module):
        def __init__(self, hid_channels, in_channels, out_channels, num_resblocks):
            super().__init__()
            self.model = nn.Sequential(
                nn.ReflectionPad2d(3),
                Downsampling(in_channels, hid_channels,
                            kernel_size=7, stride=1, padding=0, lrelu=False), # 64x256x256
                Downsampling(hid_channels, hid_channels*2, kernel_size=3, lrelu=False), # 128x128x128
                Downsampling(hid_channels*2, hid_channels*4, kernel_size=3, lrelu=False), # 256x64x64
                *[ResBlock(hid_channels*4) for _ in range(num_resblocks)], # 256x64x64
                Upsampling(hid_channels*4, hid_channels*2, kernel_size=3, output_padding=1), # 128x128x128
                Upsampling(hid_channels*2, hid_channels, kernel_size=3, output_padding=1), # 64x256x256
                nn.ReflectionPad2d(3),
                nn.Conv2d(hid_channels, out_channels, kernel_size=7, stride=1, padding=0), # 3x256x256
                nn.Tanh(),
            )
            
        def forward(self, x):
            return self.model(x)
        
def get_gen(gen_name, hid_channels, num_resblocks, in_channels=3, out_channels=3):
        if gen_name == "unet":
            return UNetGenerator(hid_channels, in_channels, out_channels)
        elif gen_name == "resnet":
            return ResNetGenerator(hid_channels, in_channels, out_channels, num_resblocks)
        else:
            raise NotImplementedError(f"Generator name '{gen_name}' not recognized.")
class Discriminator(nn.Module):
        def __init__(self, hid_channels, in_channels=3):
            super().__init__()
            self.block = nn.Sequential(
                Downsampling(in_channels, hid_channels, norm=False), # 64x128x128
                Downsampling(hid_channels, hid_channels*2), # 128x64x64
                Downsampling(hid_channels*2, hid_channels*4), # 256x32x32
                Downsampling(hid_channels*4, hid_channels*8, stride=1), # 512x31x31
                nn.Conv2d(hid_channels*8, 1, kernel_size=4, padding=1), # 1x30x30
            )
            
        def forward(self, x):
            return self.block(x)
class ImageBuffer(object):
        def __init__(self, buffer_size):
            self.buffer_size = buffer_size
            if self.buffer_size > 0:
                # the current capacity of the buffer
                self.curr_cap = 0
                # initialize buffer as empty list
                self.buffer = []
        
        def __call__(self, imgs):
            # the buffer is not used
            if self.buffer_size == 0:
                return imgs
            
            return_imgs = []
            for img in imgs:
                img = img.unsqueeze(dim=0)
                
                # fill buffer to maximum capacity
                if self.curr_cap < self.buffer_size:
                    self.curr_cap += 1
                    self.buffer.append(img)
                    return_imgs.append(img)
                else:
                    p = np.random.uniform(low=0., high=1.)
                    
                    # swap images between input and buffer with probability 0.5
                    if p > 0.5:
                        idx = np.random.randint(low=0, high=self.buffer_size)
                        tmp = self.buffer[idx].clone()
                        self.buffer[idx] = img
                        return_imgs.append(tmp)
                    else:
                        return_imgs.append(img)
            return torch.cat(return_imgs, dim=0)
class CycleGAN(L.LightningModule):
        def __init__(
            self,
            gen_name,
            num_resblocks,
            hid_channels,
            optimizer,
            lr,
            betas,
            lambda_idt,
            lambda_cycle,
            buffer_size,
            num_epochs,
            decay_epochs,
        ):
            super().__init__()
            self.save_hyperparameters(ignore=["optimizer"])
            self.optimizer = optimizer
            self.automatic_optimization = False
            
            # define generators and discriminators
            self.gen_PM = get_gen(gen_name, hid_channels, num_resblocks)
            self.gen_MP = get_gen(gen_name, hid_channels, num_resblocks)
            self.disc_M = Discriminator(hid_channels)
            self.disc_P = Discriminator(hid_channels)
            
            # initialize buffers to store fake images
            self.buffer_fake_M = ImageBuffer(buffer_size)
            self.buffer_fake_P = ImageBuffer(buffer_size)
            
        def forward(self, img):
            return self.gen_PM(img)   
                
        def init_weights(self):
            def init_fn(m):
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)
            
            for net in [self.gen_PM, self.gen_MP, self.disc_M, self.disc_P]:
                net.apply(init_fn)
            
        def setup(self, stage):
            if stage == "fit":
                self.init_weights()
                print("Model initialized.")
                
        def get_lr_scheduler(self, optimizer):
            def lr_lambdam(epoch):
                len_decay_phase = self.hparams.num_epochs - self.hparams.decay_epochs + 1.0
                curr_decay_step = max(0, epoch - self.hparams.decay_epochs + 1.0)
                val = 1.0 - curr_decay_step / len_decay_phase
                return max(0.0, val)
            
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambdam)
        
        def configure_optimizers(self):
            opt_config = {
                "lr": self.hparams.lr,
                "betas": self.hparams.betas,
            }
            opt_gen = self.optimizer(
                list(self.gen_PM.parameters()) + list(self.gen_MP.parameters()),
                **opt_config,
            )
            opt_disc = self.optimizer(
                list(self.disc_M.parameters()) + list(self.disc_P.parameters()),
                **opt_config,
            )
            optimizers = [opt_gen, opt_disc]
            schedulers = [self.get_lr_scheduler(opt) for opt in optimizers]
            return optimizers, schedulers
            
        def adv_criterion(self, y_hat, y):
            return F.mse_loss(y_hat, y)
        
        def recon_criterion(self, y_hat, y):
            return F.l1_loss(y_hat, y)
        
        def get_adv_loss(self, fake, disc):
            fake_hat = disc(fake)
            real_labels = torch.ones_like(fake_hat)
            adv_loss = self.adv_criterion(fake_hat, real_labels)
            return adv_loss
        
        def get_idt_loss(self, real, idt, lambda_cycle):
            idt_loss = self.recon_criterion(idt, real)
            return self.hparams.lambda_idt * lambda_cycle * idt_loss
        
        def get_cycle_loss(self, real, recon, lambda_cycle):
            cycle_loss = self.recon_criterion(recon, real)
            return lambda_cycle * cycle_loss
        
        def get_gen_loss(self):
            # calculate adversarial loss
            adv_loss_PM = self.get_adv_loss(self.fake_M, self.disc_M)
            adv_loss_MP = self.get_adv_loss(self.fake_P, self.disc_P)
            total_adv_loss = adv_loss_PM + adv_loss_MP
            
            # calculate identity loss
            lambda_cycle = self.hparams.lambda_cycle
            idt_loss_MM = self.get_idt_loss(self.real_M, self.idt_M, lambda_cycle[0])
            idt_loss_PP = self.get_idt_loss(self.real_P, self.idt_P, lambda_cycle[1])
            total_idt_loss = idt_loss_MM + idt_loss_PP
            
            # calculate cycle loss
            cycle_loss_MPM = self.get_cycle_loss(self.real_M, self.recon_M, lambda_cycle[0])
            cycle_loss_PMP = self.get_cycle_loss(self.real_P, self.recon_P, lambda_cycle[1])
            total_cycle_loss = cycle_loss_MPM + cycle_loss_PMP
            
            # combine losses
            gen_loss = total_adv_loss + total_idt_loss + total_cycle_loss
            return gen_loss
        
        def get_disc_loss(self, real, fake, disc):
            # calculate loss on real images
            real_hat = disc(real)
            real_labels = torch.ones_like(real_hat)
            real_loss = self.adv_criterion(real_hat, real_labels)
            
            # calculate loss on fake images
            fake_hat = disc(fake.detach())
            fake_labels = torch.zeros_like(fake_hat)
            fake_loss = self.adv_criterion(fake_hat, fake_labels)
            
            # combine losses
            disc_loss = (fake_loss + real_loss) * 0.5
            return disc_loss
        
        def get_disc_loss_M(self):
            fake_M = self.buffer_fake_M(self.fake_M)
            return self.get_disc_loss(self.real_M, fake_M, self.disc_M)
        
        def get_disc_loss_P(self):
            fake_P = self.buffer_fake_P(self.fake_P)
            return self.get_disc_loss(self.real_P, fake_P, self.disc_P)
        
        def training_step(self, batch, batch_idx):
            self.real_M = batch["monet"]
            self.real_P = batch["photo"]
            opt_gen, opt_disc = self.optimizers()

            # generate fake images
            self.fake_M = self.gen_PM(self.real_P)
            self.fake_P = self.gen_MP(self.real_M)
            
            # generate identity images
            self.idt_M = self.gen_PM(self.real_M)
            self.idt_P = self.gen_MP(self.real_P)
            
            # reconstruct images
            self.recon_M = self.gen_PM(self.fake_P)
            self.recon_P = self.gen_MP(self.fake_M)
        
            # train generators
            self.toggle_optimizer(opt_gen)
            gen_loss = self.get_gen_loss()        
            opt_gen.zero_grad()
            self.manual_backward(gen_loss)
            opt_gen.step()
            self.untoggle_optimizer(opt_gen)
            
            # train discriminators
            self.toggle_optimizer(opt_disc)
            disc_loss_M = self.get_disc_loss_M()
            disc_loss_P = self.get_disc_loss_P()
            opt_disc.zero_grad()
            self.manual_backward(disc_loss_M)
            self.manual_backward(disc_loss_P)
            opt_disc.step()
            self.untoggle_optimizer(opt_disc)
            
            # record training losses
            metrics = {
                "gen_loss": gen_loss,
                "disc_loss_M": disc_loss_M,
                "disc_loss_P": disc_loss_P,
            }
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
            
        def validation_step(self, batch, batch_idx):
            self.display_results(batch, batch_idx, "validate")
        
        def test_step(self, batch, batch_idx):
            self.display_results(batch, batch_idx, "test")
            
        def predict_step(self, batch, batch_idx):
            return self(batch)
        

        def on_train_epoch_start(self):
            # record learning rates
            curr_lr = self.lr_schedulers()[0].get_last_lr()[0]
            self.log("lr", curr_lr, on_step=False, on_epoch=True, prog_bar=True)
            
        def on_train_epoch_end(self):
            # update learning rates
            for sch in self.lr_schedulers():
                sch.step()
            
            # print current state of epoch
            logged_values = self.trainer.progress_bar_metrics
            print(
                f"Epoch {self.current_epoch+1}",
                *[f"{k}: {v:.5f}" for k, v in logged_values.items()],
                sep=" - ",
            )
            
        def on_train_end(self):
            print("Training ended.")
            
        def on_predict_epoch_end(self):
            predictions = self.trainer.predict_loop.predictions
            num_batches = len(predictions)
            batch_size = predictions[0].shape[0]
            last_batch_diff = batch_size - predictions[-1].shape[0]
            print(f"Number of images generated: {num_batches*batch_size-last_batch_diff}.")

    

