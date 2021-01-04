import torch
import torch.nn as nn
import torch.optim as optim
import constant

n_class = constant.n_class

class DCGAN_G(nn.Module):
    def __init__(self, use_upsample=True):
        super(DCGAN_G, self).__init__()

        self.generator = self.build_generator(use_upsample)

    def build_generator(self, use_upsample=True):
        concat_dim = constant.nz + constant.n_class
        ngf = constant.ngf
        if use_upsample == True:
            g = nn.Sequential(
				# Input Dim: batch_size x (concat_dim) x 1 x 1
				nn.ConvTranspose2d(concat_dim, ngf * 8, 4, 1, 0, bias=False),
				nn.BatchNorm2d(ngf * 8),
				nn.LeakyReLU(0.2, inplace=True),
				# Dim: batch_size x (num_gf * 8) x 4 x 4
				nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 4),
				nn.LeakyReLU(0.2, inplace=True),
				# Dim: batch_size x (num_gf * 4) x 8 x 8
				nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf * 2),
				nn.LeakyReLU(0.2, inplace=True),
				# Dim: batch_size x (num_gf * 2) x 16 x 16
				nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
				nn.BatchNorm2d(ngf),
				nn.LeakyReLU(0.2, inplace=True),
				# Dim: batch_size x (num_gf) x 32 x 32
				nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
				nn.Tanh()
				# Dim: batch_size x (num_channels=1) x 64 x 64 
			)
            return g
        else:
            g = nn.Sequential(
            # input is concat, going into a convolution
            nn.Conv2d(concat_dim, ngf*8, 3, 1, 1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 8) x 1 x 1
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*8, ngf*4, 3, 1, 1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 4) x 2 x 2
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*4, ngf*2, 3, 1, 1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 2) x 4 x 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf*2, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 1) x 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 1) x 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 1) x 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # (num_gf * 1) x 64 x 64
            nn.Conv2d(ngf, 1, 3, 1, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )
        return g

    def forward(self, noise, labels):
        # model input expects: batch_size * num_channel=1 * 1 * 1
        labels = labels.unsqueeze(2).unsqueeze(3)
        X = torch.cat([noise, labels], 1)
        X = self.generator(X)
        return X

class DCGAN_D(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(DCGAN_D, self).__init__()
        ndf = constant.ndf
        self.discriminator_image = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        if use_sigmoid == True:
            self.discriminator_output = nn.Sequential(
                nn.Conv2d(ndf*8+n_class, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
            self.discriminator_output = nn.Sequential(
                nn.Conv2d(ndf*8+n_class, 1, 4, 1, 0, bias=False),
            )
        
    def forward(self, images, label):
        images = self.discriminator_image(images)
        # Dim: batch_size*30 -> batch*30*4*4
        label = label.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        X = torch.cat([images, label], 1)
        X = self.discriminator_output(X)
        X = X.view(-1, 1).squeeze(1)
        return X

class Test_D(nn.Module):
    def __init__(self, use_sigmoid=True):
        super(Test_D, self).__init__()
        ndf = constant.ndf
        self.discriminator_image = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.Dropout(0.4, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.Dropout(0.4, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Dropout(0.4, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.Dropout(0.4, inplace=True),
            # state size. (ndf*8) x 4 x 4
        )

        if use_sigmoid == True:
            self.discriminator_output = nn.Sequential(
                nn.Conv2d(ndf*8+n_class, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
            self.discriminator_output = nn.Sequential(
                nn.Conv2d(ndf*8+n_class, 1, 4, 1, 0, bias=False),
            )
        
    def forward(self, images, label):
        images = self.discriminator_image(images)
        # Dim: batch_size*30 -> batch*30*4*4
        label = label.repeat(4, 4, 1, 1).permute(2, 3, 0, 1)
        X = torch.cat([images, label], 1)
        X = self.discriminator_output(X)
        X = X.view(-1, 1).squeeze(1)
        return X
    