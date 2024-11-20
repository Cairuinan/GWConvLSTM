import torch
import torch.nn as nn
from ConvLSTMCell import ConvLSTMCell
from GWConvLSTMCell import GWConvLSTMCell

def calculate_distance(filePath):

    dataset = gdal.Open(filePath)  # Open tif

    adfGeoTransform = dataset.GetGeoTransform()  # Read geographic information

    nXSize = dataset.RasterXSize
    nYSize = dataset.RasterYSize

    arrSlope = []  # Used to save the (X, Y) coordinates of each pixel
    for i in range(nYSize):
        row = []
        for j in range(nXSize):
            px = adfGeoTransform[0] + i * adfGeoTransform[1] + j * adfGeoTransform[2]
            py = adfGeoTransform[3] + i * adfGeoTransform[4] + j * adfGeoTransform[5]
            col = [px, py]
            row.append(col)
        arrSlope.append(row)

    arr = np.array(arrSlope)

    lon = arr[:, :, 0].reshape((-1, 1))
    lat = arr[:, :, 1].reshape((-1, 1))
    coords = np.concatenate((lon, lat), axis=1)

    # calculate distance matrix
    dist = distance.cdist(coords, coords, 'euclidean')

    x = dist.astype(np.float32)

    distance_matrix = torch.tensor(x).cuda()
    distance_matrix.to(torch.float32)

    return distance_matrix

class RNN(nn.Module):
    def __init__(self, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.num_layers = configs.num_layers
        self.w_num_layers = configs.w_num_layers

        # Define the prediction network
        cell_list = []
        for i in range(num_layers):
            in_channel = configs.frame_channel if i == 0 else configs.num_hidden
            cell_list.append(
                GWConvLSTMCell(in_channel, configs.num_hidden, configs.img_height, configs.img_width, configs.filter_size,
                                configs.stride)
            )

        self.cell_list = nn.ModuleList(cell_list)

        # Define the spatial weight generation network
        w_cell_list = []

        for i in range(self.w_num_layers):
            w_cell_list.append(
                ConvLSTMCell(configs.num_hidden, configs.num_hidden, configs.img_height, configs.img_width, configs.filter_size,
                                configs.stride, configs.device)
            )

        self.w_cell_list = nn.ModuleList(w_cell_list)

        # Calculate distance matrix
        self.distance = calculate_distance(r'E:\02GWConvLSTM\data\ndvi\train\000.tif')

        self.conv_last = nn.Conv2d(configs.num_hidden, self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames):
        # [batch, seq, channel, height, width]
        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # Define a fully connected linear layer to make the distance matrix conform to the input shape
        fc = torch.nn.Linear(height * width, batch * self.configs.num_hidden, bias=True).to(self.configs.device)
        w_t = fc(self.distance)
        w_t_begin = w_t.view(batch, self.configs.num_hidden, height, width)

        next_frames = []
        h_t = []
        c_t = []
        w_t = []
        w_h = []
        w_c = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.configs.num_hidden, height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        for s in range((self.configs.total_length - 1) * self.w_num_layers):
            zeros_w = torch.zeros([batch, self.configs.num_hidden, height, width]).to(self.configs.device)
            w_t.append(zeros_w)
            w_h.append(zeros_w)
            w_c.append(zeros_w)

        for t in range(self.configs.total_length - 1):

            if t < self.configs.input_length:
                net = frames[:, t]
                w_t[0] = w_t_begin
            else:
                net = x_gen

            # Spatial weight generate
            if t == 0:
                w_h[0], w_c[0] = self.w_cell_list[0](w_t[0], w_h[0], w_c[0])
                for w_layer in range(1, self.w_num_layers):
                    w_h[t + w_layer], w_c[t + w_layer] = self.w_cell_list[w_layer](w_h[t + w_layer - 1], w_h[t + w_layer], w_c[t + w_layer])
            else:
                w_h[t * self.w_num_layers], w_c[t * self.w_num_layers] = self.w_cell_list[0](w_h[t * self.w_num_layers-1], w_h[t * self.w_num_layers], w_c[t * self.w_num_layers])
                for w_layer in range(1, self.w_num_layers):
                    w_h[t * self.w_num_layers + w_layer], w_c[t * self.w_num_layers + w_layer] = self.w_cell_list[w_layer](w_h[t * self.w_num_layers + w_layer- 1], w_h[t * self.w_num_layers+ w_layer],
                                                                             w_c[t * self.w_num_layers + w_layer])

            # Prediction network
            h_t[0], c_t[0] = self.cell_list[0](net, h_t[0], c_t[0], w_h[(t+1) * self.w_num_layers-1])

            for i in range(1, self.num_layers):
                
                h_t[i], c_t[i] = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], w_h[(t+1) * self.w_num_layers-1])

            # Match the final outputs dimension to the number of input features
            x_gen = self.conv_last(h_t[self.num_layers - 1])

            next_frames.append(x_gen)

        next_frames = torch.stack(next_frames, dim=1).contiguous()
        
        return next_frames