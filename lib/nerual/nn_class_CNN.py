import torch.nn as nn
import torch.nn.functional as F
import wandb

class Network_l3(nn.Module):

    def __init__(self, traceLen, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Network_l3, self).__init__()
        '''branch 1: traces compressing'''
        # traceLen, bitLen = input_size

        self.num_classes = num_classes
        self.traceLen = traceLen
        self.conv1_inchannel = 1
        self.conv1_outchannel = 4

        self.conv2_inchannel = 4
        self.conv2_outchannel = 16

        self.conv3_inchannel = 16
        self.conv3_outchannel = 32




        self.cov2fc_H = int( (traceLen[0]-wandb.config.kernel+1)/2 )
        #self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-wandb.config.kernel+1)/2 )
        self.cov2fc_H = int( (self.cov2fc_H-wandb.config.kernel+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-wandb.config.kernel+1)/2 )
        #self.cov2fc_W = int( (self.cov2fc_W-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-wandb.config.kernel_width+1)/2 )
        self.cov2fc_W = int( (self.cov2fc_W-wandb.config.kernel_width+1)/2 )


        # self.conv_2L = int( (self.conv_1L)/2 )
        
        self.feedback = None


        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=wandb.config.kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(wandb.config.kernel,wandb.config.kernel_width), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(wandb.config.kernel,wandb.config.kernel_width), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3_outchannel)


        # self.fc0 = nn.Linear(self.traceLen, 100)
        self.dt0 = nn.Dropout(0.5)
        # self.dt0 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, num_classes)
        
        self.fc1 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, 9)

        self.fc2 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, 64)
        self.fc3 = nn.Linear(64,9)


    def forward(self, x):
        # x1, x2 = x
        '''branch 1: traces compressing'''
        # out = F.relu(self.fc0(x))
        # self.feedback = torch.zeros_like( x )

        # for loop in range(2):

            # out = [x, self.feedback]
            # out = torch.cat( out, 1 )
        
        out1 = F.selu(self.bn1(self.conv1(x)))
        
        out1 = F.max_pool2d(out1, 2)

        #out1 = F.relu(self.bn2(self.conv2(out1)))
        #out1 = F.max_pool2d(out1, 2)

        out1 = F.selu(self.bn2(self.conv2(out1)))
        out1 = F.max_pool2d(out1, 2)

        out1 = F.selu(self.bn3(self.conv3(out1)))
        out1 = F.max_pool2d(out1, 2)



        out1 = out1.view(out1.size(0), -1)


        out1 = self.dt0(out1)
        
        if(wandb.config.dense==2):
            out1 = F.relu(self.fc2(out1))
            out1 = F.softmax(self.fc3(out1), dim=1)
        else:
            out1 = F.softmax(self.fc1(out1), dim=1)

        '''branch 2: bits compressing'''
        # out2 = F.selu(self.rfc1(x2))


        return out1





class Network_l2(nn.Module):

    def __init__(self, traceLen, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Network_l2, self).__init__()
        '''branch 1: traces compressing'''
        # traceLen, bitLen = input_size

        self.num_classes = num_classes
        self.traceLen = traceLen
        self.conv1_inchannel = 1
        self.conv1_outchannel = 4

        self.conv2_inchannel = 4
        self.conv2_outchannel = 16

        self.conv3_inchannel = 16
        self.conv3_outchannel = 32




        self.cov2fc_H = int( (traceLen[0]-wandb.config.kernel+1)/2 )
        #self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-wandb.config.kernel+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-wandb.config.kernel+1)/2 )
        #self.cov2fc_W = int( (self.cov2fc_W-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-wandb.config.kernel_width+1)/2 )


        # self.conv_2L = int( (self.conv_1L)/2 )
        
        self.feedback = None


        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=wandb.config.kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(wandb.config.kernel,wandb.config.kernel_width), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(wandb.config.kernel,wandb.config.kernel_width), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3_outchannel)


        # self.fc0 = nn.Linear(self.traceLen, 100)
        self.dt0 = nn.Dropout(0.5)
        # self.dt0 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, num_classes)

        self.fc1 = nn.Linear(self.conv2_outchannel*self.cov2fc_W*self.cov2fc_H, 9)

        self.fc2 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, 64)
        self.fc3 = nn.Linear(64,9)


    def forward(self, x):
        # x1, x2 = x
        '''branch 1: traces compressing'''
        # out = F.relu(self.fc0(x))
        # self.feedback = torch.zeros_like( x )

        # for loop in range(2):

            # out = [x, self.feedback]
            # out = torch.cat( out, 1 )
        
        out1 = F.selu(self.bn1(self.conv1(x)))
        
        out1 = F.max_pool2d(out1, 2)

        #out1 = F.relu(self.bn2(self.conv2(out1)))
        #out1 = F.max_pool2d(out1, 2)

        out1 = F.selu(self.bn2(self.conv2(out1)))
        out1 = F.max_pool2d(out1, 2)



        out1 = out1.view(out1.size(0), -1)


        out1 = self.dt0(out1)
        
        if(wandb.config.dense==2):
            out1 = F.relu(self.fc2(out1))
            out1 = F.softmax(self.fc3(out1), dim=1)
        else:
            out1 = F.softmax(self.fc1(out1), dim=1)

        '''branch 2: bits compressing'''
        # out2 = F.selu(self.rfc1(x2))


        return out1



class Network_l3_u(nn.Module):

    def __init__(self, traceLen, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Network_l3_u, self).__init__()
        '''branch 1: traces compressing'''
        # traceLen, bitLen = input_size

        self.num_classes = num_classes
        self.traceLen = traceLen
        self.conv1_inchannel = 1
        self.conv1_outchannel = 4

        self.conv2_inchannel = 4
        self.conv2_outchannel = 16

        self.conv3_inchannel = 16
        self.conv3_outchannel = 32




        self.cov2fc_H = int( (traceLen[0]-wandb.config.kernel+1))
        #self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-wandb.config.kernel+1)/2 )
        self.cov2fc_H = int( (self.cov2fc_H-wandb.config.kernel+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-wandb.config.kernel+1) )
        #self.cov2fc_W = int( (self.cov2fc_W-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-wandb.config.kernel_width+1)/2 )
        self.cov2fc_W = int( (self.cov2fc_W-wandb.config.kernel_width+1)/2 )


        # self.conv_2L = int( (self.conv_1L)/2 )
        
        self.feedback = None


        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=wandb.config.kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(wandb.config.kernel,wandb.config.kernel_width), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(wandb.config.kernel,wandb.config.kernel_width), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3_outchannel)


        # self.fc0 = nn.Linear(self.traceLen, 100)
        self.dt0 = nn.Dropout(0.5)
        # self.dt0 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, num_classes)

        self.fc1 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, 9)
        self.fc2 = nn.Linear(self.conv3_outchannel*self.cov2fc_W*self.cov2fc_H, 64)
        self.fc3 = nn.Linear(64,9)


    def forward(self, x):
        # x1, x2 = x
        '''branch 1: traces compressing'''
        # out = F.relu(self.fc0(x))
        # self.feedback = torch.zeros_like( x )

        # for loop in range(2):

            # out = [x, self.feedback]
            # out = torch.cat( out, 1 )
        
        out1 = F.selu(self.bn1(self.conv1(x)))
        

        #out1 = F.relu(self.bn2(self.conv2(out1)))
        #out1 = F.max_pool2d(out1, 2)

        out1 = F.selu(self.bn2(self.conv2(out1)))
        out1 = F.max_pool2d(out1, 2)

        out1 = F.selu(self.bn3(self.conv3(out1)))
        out1 = F.max_pool2d(out1, 2)



        out1 = out1.view(out1.size(0), -1)


        out1 = self.dt0(out1)
        

        if(wandb.config.dense==2):
            out1 = F.relu(self.fc2(out1))
            out1 = F.softmax(self.fc3(out1), dim=1)
        else:
            out1 = F.softmax(self.fc1(out1), dim=1)

        '''branch 2: bits compressing'''
        # out2 = F.selu(self.rfc1(x2))


        return out1



class Network_jc(nn.Module):

    def __init__(self, traceLen, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Network_jc, self).__init__()
        '''branch 1: traces compressing'''
        # traceLen, bitLen = input_size

        self.num_classes = num_classes
        self.traceLen = traceLen
        self.conv1_inchannel = 1
        self.conv1_outchannel = 4

        self.conv2_inchannel = 4
        self.conv2_outchannel = 8

        self.conv3_inchannel = 8
        self.conv3_outchannel = 16

        # self.conv4_inchannel = 32
        # self.conv4_outchannel = 32
        # self.convfb1_inchannel = 128
        # self.convfb1_outchannel = 1


        # each channel output size: when strike equals to 1, first pooling size is 3, second pooling size equals to 4
        # self.cov2fc = int( ((self.traceLen - 256 + 1)/2 - 128 + 1)/4 )
        self.cov2fc_H = int( (traceLen[0]-3+1) )
        self.cov2fc_H = int( (self.cov2fc_H-3+1) )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-16+1) )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-16+1)/2 )



        self.feedback = None
        # ascad paper profiling network
        # self.conv1 = nn.Conv1d(1, 64, kernel_size=11, stride=1, padding=5)
        # self.conv2 = nn.Conv1d(64, 128, kernel_size=11, stride=1, padding=5)
        # self.conv3 = nn.Conv1d(128, 256, kernel_size=11, stride=1, padding=5)
        # self.conv4 = nn.Conv1d(256, 512, kernel_size=11, stride=1, padding=5)
        # self.conv5 = nn.Conv1d(512, 512, kernel_size=11, stride=1, padding=5)

        # self.fc1 = nn.Linear(10752, 2048)
        # self.fc2 = nn.Linear(2048, 2048)
        # self.fc3 = nn.Linear(2048, num_classes)

        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=(3,16), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(3,16), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(3,16), stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3_outchannel)
        # # self.bn3 = BatchInstanceNorm1d(num_features=self.conv3_outchannel)
        # self.conv4 = nn.Conv1d(self.conv4_inchannel, self.conv4_outchannel, kernel_size=16, stride=1, padding=0)
        # self.bn4 = nn.BatchNorm1d(num_features=self.conv4_outchannel)

        # self.conpad1 = nn.ConstantPad1d( 150, 0 )

        # self.convfb1 = nn.Conv1d(self.convfb1_inchannel, self.convfb1_outchannel, kernel_size=3, stride=1, padding=1)
        # self.bnfb1 = nn.BatchNorm1d(num_features=self.convfb1_outchannel)

        # self.fc0 = nn.Linear(self.traceLen, 100)
       
        self.dt0 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(self.conv3_outchannel*self.cov2fc_H*self.cov2fc_W, 128)
        self.dt1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, num_classes)
        # self.fc3 = nn.Linear(32, num_classes)
        
        '''branch 2: bits compressing'''
        # self.bitLen = bitLen
        # self.rfc1 = nn.Linear(self.bitLen, num_classes)


    def forward(self, x):
        # x1, x2 = x
        '''branch 1: traces compressing'''
        # out = F.relu(self.fc0(x))
        # self.feedback = torch.zeros_like( x )

        # for loop in range(2):

            # out = [x, self.feedback]
            # out = torch.cat( out, 1 )
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = F.selu(out1)
        # out1 = F.selu(self.bn1(self.conv1(x)))
        # out1 = F.max_pool1d(out1, 2)

        out1 = F.selu(self.bn2(self.conv2(out1)))
        # out1 = F.max_pool2d(out1, 2)

        out1 = F.selu(self.bn3(self.conv3(out1)))
        # out1 = F.selu(self.bn4(self.conv4(out1)))
        out1 = F.max_pool2d(out1, 2)

            # if ( loop == 0 ):


        out1 = out1.view(out1.size(0), -1)
        out1 = self.dt0(out1)
        out1 = F.selu(self.fc1(out1))
        out1 = self.dt1(out1)
        out1 = self.fc2(out1)
        # # out = out.mean(2)
        # out = self.fc3(out)
        '''branch 2: bits compressing'''
        # out2 = F.selu(self.rfc1(x2))


        return out1




class mlp(nn.Module):

    def __init__(self, traceLen, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(mlp, self).__init__()
        '''branch 1: traces compressing'''
        # traceLen, bitLen = input_size

        self.num_classes = num_classes

       
        self.dt0 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(traceLen[1]*traceLen[0], wandb.config.channel_1)
        self.dt1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(wandb.config.channel_1, 1)
        self.fc3 = nn.Linear(10, 1)
        
        # self.fc3 = nn.Linear(32, num_classes)
        
        '''branch 2: bits compressing'''
        # self.bitLen = bitLen
        # self.rfc1 = nn.Linear(self.bitLen, num_classes)


    def forward(self, x):
        # x1, x2 = x



 
        out1 = x.view(x.size(0), -1)
        out1 = F.selu(self.fc1(out1))
        out1 = F.selu(self.fc2(out1))
        #out1 = F.selu(self.fc3(out1))
        # # out = out.mean(2)
        # out = self.fc3(out)
        '''branch 2: bits compressing'''
        # out2 = F.selu(self.rfc1(x2))


        return out1


import torch.nn as nn
import torch.nn.functional as F


class mlp_jc(nn.Module):

    def __init__(self, traceLen, num_classes):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(mlp_jc, self).__init__()
        '''branch 1: traces compressing'''

        self.fc1 = nn.Linear(traceLen, 128)
        # self.dt1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, num_classes)


    def forward(self, x):

        out1 = F.selu(self.fc1(x))
        # out1 = self.dt1(out1)
        out1 = self.fc2(out1)
        out1 = self.fc3(out1)
 



        return out1