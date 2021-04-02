
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from aspp3d import ASPP
import time

# import sparseconvnet as scn

def upsample(x, scale):
    ''' 3d upsample (trt need it)'''
    # x_shape = torch.tensor(x.shape)
    # dimx = int(x_shape[2] * scale)
    # dimy = int(x_shape[3] * scale)
    # dimz = int(x_shape[4] * scale)
    # return F.interpolate(x, size=(dimx, dimy, dimz), mode='nearest')
    sh = torch.tensor(x.shape)
    return F.interpolate(x, size=(int(sh[2]*scale), int(sh[3]*scale), int(sh[4]*scale)), mode='nearest')

def count_num_model_params(model):
    num = 0
    for p in list(model.parameters()):
        cur = 1
        for s in list(p.size()):
            cur = cur * s
        num += cur
    return num

FSIZE0 = 3
FSIZE1 = 2
ACTIVATE = nn.LeakyReLU(0.2, inplace=True)
FIX = False

def bridge_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

class conv_block(nn.Module):
    ''' down sample bloc'''
    def __init__(self, in_dim, out_dim, use_bias=False):
        super(conv_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size=2, stride=2, padding=0, bias=True),
            # nn.Conv3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, bias=True),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class upconv_block(nn.Module):
    ''' up sample bloc'''

    def __init__(self, in_dim, out_dim, stride=1, use_bias=False):
        super(upconv_block, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

 
# def vgg_block(self, inplanes, planes, stride=1, use_bias=False):
#     block = nn.Sequential(
#         nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=use_bias),
#         nn.BatchNorm3d(planes),
#         nn.ReLU(inplace=True),
#     )
#     return block


class UNet(nn.Module):
    '''
    Suppose reps = [3,3,3]
    '''
    def __init__(self, nPlanes, reps):
        super(UNet, self).__init__()
        assert reps == 1
        assert len(nPlanes) == 3

        # self.up = nn.Upsample(scale_factor=2)

        self.res1 = conv_block(nPlanes[0], nPlanes[1])
        # self.bn1 = nn.BatchNorm3d(nPlanes[0])
        self.res2 = conv_block(nPlanes[1], nPlanes[2])
        # self.bn2 = nn.BatchNorm3d(nPlanes[1])
        self.bridge = bridge_block(nPlanes[2], nPlanes[2])

        self.bn_relu = nn.Sequential(
            nn.BatchNorm3d(nPlanes[2]+nPlanes[1]+nPlanes[0]),
            nn.ReLU(inplace=True)           
        )

        # self.up1 = upconv_block(nPlanes[2], nPlanes[2])
        # self.up2 = upconv_block(nPlanes[2]+nPlanes[1], nPlanes[2]+nPlanes[1])
        # self.up2 = upconv_block(nPlanes[2]+nPlanes[1]+nPlanes[0], nPlanes[2]+nPlanes[1]+nPlanes[0])

    def forward(self, x):
        x0 = x
        x1 = self.res1(x0)
        x = self.res2(x1)
        x = self.bridge(x)

        x = upsample(x, 2)
        # print("fuck 1:", x.shape)
        # dimx, dimy, dimz = x.size()[2] * 2, x.size()[3] * 2, x.size()[4] * 2
        # x = F.interpolate(x, size=(int(dimx), int(dimy), int(dimz)), mode='nearest')
        x = torch.cat((x, x1), dim=1)

        dimx, dimy, dimz = x.size()[2] * 2, x.size()[3] * 2, x.size()[4] * 2
        x = F.interpolate(x, size=(int(dimx), int(dimy), int(dimz)), mode='nearest')
        print("fuck 2:", x.shape)
        # x = F.interpolate(x, size=(80, 80, 40), mode='nearest')

        x = torch.cat((x, x0), dim=1)

        return self.bn_relu(x)

class PreEncoderLayer(nn.Module):
    ''' sparse encoder layer:
    p0: dense input to sparse
    p1: conv, input channels to output channels, f=3
    p2: conv, save output as feature2
    p3: conv, downsample by 2, save output as feature3 (if p4)
    p4: sparse to dense 
    TODO: detail of scn.InputLayer & scn.SparseToDense
    '''
    def __init__(self, nf_in, nf):
        nn.Module.__init__(self)
        data_dim = 3
        self.nf_in = nf_in
        self.nf = nf
        self.use_bias = False

        self.p1 = nn.Conv3d(nf_in, nf, kernel_size=FSIZE0, stride=1, padding=1, bias=self.use_bias)

        self.p2 = nn.Sequential(
            nn.BatchNorm3d(nf),
            nn.ReLU(True),
            nn.Conv3d(nf, nf, kernel_size=FSIZE0, stride=1, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf),
            nn.ReLU(True),
            nn.Conv3d(nf, nf, kernel_size=FSIZE0, stride=1, padding=1, bias=self.use_bias),
        )

        self.p2_2 = nn.Sequential(
            nn.BatchNorm3d(nf),
            nn.ReLU(True),
        )

        self.p3 = nn.Sequential(
            nn.Conv3d(nf, nf, kernel_size=FSIZE1, stride=2, padding=0, bias=self.use_bias),
            nn.BatchNorm3d(nf),
            nn.ReLU(True)
        )

    def forward(self,x):
        x0 = self.p1(x)
        x = self.p2(x0) + x0
        x = self.p2_2(x)
        ft2 = x
        x = self.p3(x)
        return x, [ft2]


class TSDFEncoder(nn.Module):
    '''
    sparse conv layers: 4, decrease resolution by 2 for 3 times, input is sparse, last layer's output is dense
    dense encoder-decoder: 16-8-4-4-8-16, with skip contacts, output is dense
    occ + sdf pred: take the output of dense-decoder, use 1*1*1 conv to output occ / sdf prediction
    '''
    def __init__(self, nf_in, nf_per_level, nf_out, use_skip_sparse=True, use_skip_dense=True):
        nn.Module.__init__(self)
        assert (type(nf_per_level) is list)
        data_dim = 3
        self.use_skip_sparse = use_skip_sparse
        self.use_skip_dense = use_skip_dense
        self.use_aspp = False
        #self.use_bias = True
        self.use_bias = False
        modules = []

        for level in range(len(nf_per_level)):
            nf_in = nf_in if level == 0 else nf_per_level[level-1]
            modules.append(PreEncoderLayer(nf_in, nf_per_level[level]))
        self.process_sparse = nn.Sequential(*modules)
        # print("TSDF SPARSE: ", self.process_sparse)
        nf = nf_per_level[-1]
        # 16 -> 8
        nf0 = nf*3 // 2
        self.encode_dense0 = nn.Sequential(
            nn.Conv3d(nf, nf0, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf0),
            nn.ReLU(True)
        )

        nf1 = nf*2

        # 8 -> 4
        self.encode_dense1 = nn.Sequential(
            nn.Conv3d(nf0, nf1, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf1),
            nn.ReLU(True)
        )

        # 4 -> 4
        nf2 = nf1
        self.bottleneck_dense2 = nn.Sequential(
            nn.Conv3d(nf1, nf2, kernel_size=1, bias=self.use_bias),
            nn.BatchNorm3d(nf2),
            nn.ReLU(True)
        )
        # 4 -> 8
        nf3 = nf2 if not self.use_skip_dense else nf1+nf2
        nf4 = nf3 // 2
        self.decode_dense3 = nn.Sequential(
            nn.ConvTranspose3d(nf3, nf4, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf4),
            nn.ReLU(True)
        )
        # 8 -> 16
        if self.use_skip_dense:
            nf4 += nf0
        nf5 = nf4 // 2
        self.decode_dense4 = nn.Sequential(
            nn.ConvTranspose3d(nf4, nf5, kernel_size=4, stride=2, padding=1, bias=self.use_bias),
            nn.BatchNorm3d(nf5),
            nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Conv3d(nf5, nf_out, kernel_size=1, bias=self.use_bias),
            nn.BatchNorm3d(nf_out),
            nn.ReLU(True)
        )
        # occ prediction
        self.occpred = nn.Sequential(
            nn.Conv3d(nf_out, 1, kernel_size=1, bias=self.use_bias)
        )

        # debug stats
        # params_encodesparse = count_num_model_params(self.process_sparse)
        # params_encodedense = count_num_model_params(self.encode_dense0) + count_num_model_params(self.encode_dense0) + count_num_model_params(self.encode_dense1) + count_num_model_params(self.bottleneck_dense2)
        # params_decodedense = count_num_model_params(self.decode_dense3) + count_num_model_params(self.decode_dense4) + count_num_model_params(self.final) + count_num_model_params(self.occpred)
        # print('[TSDFEncoder] params encode sparse', params_encodesparse)
        # print('[TSDFEncoder] params encode dense', params_encodedense)
        # print('[TSDFEncoder] params decode dense', params_decodedense)

    def forward(self,x):
        feats_sparse = []
        for k in range(len(self.process_sparse)):
            x, ft = self.process_sparse[k](x)
            if self.use_skip_sparse:
                feats_sparse.extend(ft)
        feats_sparse.append(x)

        # print("before encode: ", x.shape)


        enc0 = self.encode_dense0(x)
        if self.use_aspp:
            enc_aspp = self.aspp(enc0)
            enc1 = self.encode_dense1(enc_aspp)
        else:
            enc1 = self.encode_dense1(enc0)

        # print("enc1: ", enc1.shape)

        bottleneck = self.bottleneck_dense2(enc1)
        if self.use_skip_dense:
            dec0 = self.decode_dense3(torch.cat([bottleneck, enc1], 1))
        else:
            dec0 = self.decode_dense3(bottleneck)

        # print("dec0: ", dec0.shape)

        if self.use_skip_dense:
            x = self.decode_dense4(torch.cat([dec0, enc0], 1))
        else:
            x = self.decode_dense4(dec0)
        x = self.final(x)
        occ = self.occpred(x).repeat([1,2,1,1,1])
        return x, occ, feats_sparse

class Refinement(nn.Module):
    '''
    to_next_level_locs: up sample, locs: x,y,z *= 2, extend 1 grid into 8 grids 
    geo filter: mast = Sigmoid(occ) > 0.5
    occ, sdf: use a nn.Linear to go trough 16 channels (into 1 channel)
    TODO: why p and n? why FullyConvolutionalNet? why Input / Output layers(for upsampling & Linear?)?
    '''
    def __init__(self, nf_in, nf, truncation=3, pass_occ=True, pass_feats=True):
        nn.Module.__init__(self)
        data_dim = 3
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        self.nf_in = nf_in
        self.nf = nf
        self.truncation = truncation
        self.use_aspp = True

        self.p1 = nn.Sequential(
            nn.ConvTranspose3d(nf_in, nf, kernel_size=FSIZE0, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(nf),
            nn.ReLU(True)
        )

        if self.use_aspp:
            self.p2 = ASPP(nf, nf) 
            self.p3 = nn.Sequential(
            nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(nf),
            nn.ReLU(inplace=True)
        )
        else:
            self.p2 = UNet(nPlanes=[nf, nf, nf], reps=1)

            self.p3 = nn.Sequential(
                nn.Conv3d(nf*3, nf, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(nf),
                nn.ReLU(inplace=True)
            )

        # self.p3 = nn.Sequential(
        #     nn.Conv3d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm3d(nf),
        #     nn.ReLU(inplace=True)
        # )        

        self.occ = nn.Conv3d(nf, 1, kernel_size=1, stride=1, padding=0, bias=False)
        # self.occ = nn.Conv3d(nf, 1, kernel_size=3, stride=2, padding=1, bias=self.use_bias)
        # self.sdf = nn.Conv3d(nf, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self,x):

        x = self.p1(x)
        x = self.p2(x)
        # x = upsample(x, 2)
        # print("fuck 3:", x.shape)
        dimx, dimy, dimz = x.size()[2] * 2, x.size()[3] * 2, x.size()[4] * 2
        x = F.interpolate(x, size=(int(dimx), int(dimy), int(dimz)), mode='nearest')


        x = self.p3(x)

        occ = self.occ(x)
        # mask out for next level processing
        # mask = (nn.Sigmoid()(occ) < 0.5)
        # #print('x', x.type(), x.shape, torch.min(x).item(), torch.max(x).item())
        # x[mask.repeat((1,x.shape[1],1,1,1))] *= 0
        # sdf[mask] *= 0
        # if self.pass_feats and self.pass_occ:

        # elif self.pass_feats:
        #     feats = x
        # elif self.pass_occ:
        #     feats = out
        return None, occ

# ==== model ==== #
class GenModel(nn.Module):
    def __init__(self, encoder_dim, input_nf, nf_coarse, nf, num_hierarchy_levels, pass_occ, pass_feats, use_skip_sparse, use_skip_dense, truncation=3):
        nn.Module.__init__(self)
        self.truncation = truncation
        self.pass_occ = pass_occ
        self.pass_feats = pass_feats
        self.max_pool = nn.MaxPool3d(kernel_size=2)
        # encoder
    
        # encoder
        self.encoder = TSDFEncoder(1, [8], 16, True, True)

        # self.refinement = Refinement(26, 16, truncation=self.truncation, pass_occ=True, pass_feats=True)
        self.refinement = nn.Sequential(Refinement(26, 16, truncation=self.truncation, pass_occ=True, pass_feats=True))
        
    def forward(self, x):

        loss_weights = [1.0,1.0,1.0,1.0]
        if len(x.shape) == 4:
            # x.unsqueeze_(1)
            x = x.unsqueeze(1)
        init_input = x
        init_known_mask = (init_input >= 0).squeeze(1)
        init_input_1 = self.max_pool(init_input)
        init_known_mask_1 = (init_input_1 >= 0).squeeze(1)

        # x = torch.cat((x, init_known_mask.float()), 1)

        outputs = []
        #print('[model] x', x[0].shape, x[1].shape, torch.max(x[0][:,0]).item(), torch.max(x[0][:,1]).item(), torch.max(x[0][:,2]).item())
        # encode
        x, out, feats_dense = self.encoder(x)
        batch_size = x.shape[0]

        outputs.append(out)
        #print('locs, feats', locs.shape, locs.type(), feats.shape, feats.type(), x.shape)
        #raw_input('sdflkj')
        x = torch.cat((x, out), 1)
        # fix conflit in occ layer
        # if FIX:
        #     x[:,-2][init_known_mask_1] = init_input_1[:, 0][init_known_mask_1]

        # if self.use_skip_sparse:
        x = torch.cat((feats_dense[len(self.refinement)-0], x), 1)
        x, out = self.refinement[0](x)
        # embed()
        return out[:,0]
        # return 

if __name__ == '__main__':
    use_cuda = True

    model = GenModel(encoder_dim=8, input_nf=1, nf_coarse=16, nf=16, num_hierarchy_levels=2, pass_occ=True, pass_feats=True, use_skip_sparse=1, use_skip_dense=1)
    # model = GenModel_fix(input_dim=(128,128,128), input_nf=1, nf_coarse=16, nf=16, num_hierarchy_levels=2, pass_occ=True, pass_feats=True, use_skip_sparse=1, use_skip_dense=1)
    # exit()
    # batch size 10 -- all batches are identical
    input = torch.rand(1,80,80,40)
    print('input', input.shape, torch.min(input).item(), torch.max(input).item())
    if use_cuda:
        model = model.cuda()
        input = input.cuda()
    model.eval()
    ts = time.time()
    for i in range(10):
        output_occ = model(input)
    print("time: ", (time.time() - ts) / 1000)
    # print('output_sdf', output_sdf.shape, torch.min(output_sdf).item(), torch.max(output_sdf).item())
    print('output_occs', output_occ.shape)

