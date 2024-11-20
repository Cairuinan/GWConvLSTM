import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, height, filter_size, stride, device):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, height])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, height])
        )

        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, height)).to(device)
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, height)).to(device)
        self.Wcg = nn.Parameter(torch.zeros(1, num_hidden, width, height)).to(device)
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, height)).to(device)

    def forward(self, x_t, h_t, c_t):

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)

        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)

        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        sig1 = i_x + i_h + self.Wci * c_t

        i_t = torch.sigmoid(sig1)

        sig2 = f_x + f_h + self.Wcf * c_t + self._forget_bias

        f_t = torch.sigmoid(sig2)

        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        sig3 = o_x + o_h + self.Wco * c_t

        o_t = torch.sigmoid(sig3)

        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new
