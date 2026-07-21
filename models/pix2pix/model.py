import torch
import torch.nn as nn

class UNetSkipConnectionBlock(nn.Module):
    """Helper class to build U-Net recursively with skip connections."""
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, 
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
            
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            # Skip connection by concatenating along channel dimension (dim=1)
            return torch.cat([x, self.model(x)], 1)


class UNetGenerator(nn.Module):
    """U-Net generator with skip connections."""
    def __init__(self, input_nc=2, output_nc=3, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=True):
        """
        input_nc: number of input channels (2 for IR: B10, B11)
        output_nc: number of output channels (3 for RGB)
        num_downs: number of downsamplings in U-Net (e.g., 8 for 256x256 image size)
        ngf: number of filters in the last conv layer
        """
        super(UNetGenerator, self).__init__()
        # Construct U-Net recursively from the innermost block to the outermost block
        # Innermost block
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, innermost=True, norm_layer=norm_layer)
        
        # Intermediate blocks
        for _ in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
            
        # Gradually reduce channels back
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        
        # Outermost block
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN Discriminator (70x70 receptive field)."""
    def __init__(self, input_nc=5, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """
        input_nc: number of channels of input + target concatenated (2 for IR + 3 for RGB = 5 channels)
        ndf: number of filters in the first conv layer
        n_layers: number of intermediate conv layers
        """
        super(PatchGANDiscriminator, self).__init__()
        
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Output layer maps to a 1-channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
