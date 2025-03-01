import torch.nn as nn

class MHCA(nn.Module):
    def __init__(self, n_feats, ratio):
        super(MHCA, self).__init__()

        out_channels = int(n_feats // ratio)

        head_1 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(in_channels=out_channels, out_channels=n_feats, kernel_size=1, padding=0, bias=True)
        ]

        head_2 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=3, padding=1, bias=True)
        ]

        head_3 = [
            nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=5, padding=2, bias=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=5, padding=2, bias=True)
        ]

        # head_4 = [
        #     nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=7, padding=3, bias=True),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=7, padding=3, bias=True)
        # ]

        # head_5 = [
        #     nn.Conv2d(in_channels=n_feats, out_channels=out_channels, kernel_size=9, padding=4, bias=True),
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(in_channels=out_channels, out_channels=n_feats, kernel_size=9, padding=4, bias=True)
        # ]

        self.head_1 = nn.Sequential(*head_1)
        self.head_2 = nn.Sequential(*head_2)
        self.head_3 = nn.Sequential(*head_3)
        #self.head_4 = nn.Sequential(*head_4)
        #self.head_5 = nn.Sequential(*head_5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res_h1 = self.head_1(x)
        res_h2 = self.head_2(x)
        res_h3 = self.head_3(x)
        #res_h4 = self.head_4(x)
        #res_h5 = self.head_5(x)
        m_c = self.sigmoid(res_h1 + res_h2 + res_h3 )#+ res_h4 + res_h5)
        res = x * m_c
        return res