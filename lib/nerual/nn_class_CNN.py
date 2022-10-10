import torch.nn as nn
import torch.nn.functional as F
from wandb_config import wandb_config

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




        self.cov2fc_H = int( (traceLen[0]-wandb_config.kernel+1)/2 )
        #self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-wandb_config.kernel+1)/2 )
        self.cov2fc_H = int( (self.cov2fc_H-wandb_config.kernel+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-wandb_config.kernel+1)/2 )
        #self.cov2fc_W = int( (self.cov2fc_W-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-wandb_config.kernel_width+1)/2 )
        self.cov2fc_W = int( (self.cov2fc_W-wandb_config.kernel_width+1)/2 )


        # self.conv_2L = int( (self.conv_1L)/2 )
        
        self.feedback = None


        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=wandb_config.kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(wandb_config.kernel,wandb_config.kernel_width), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(wandb_config.kernel,wandb_config.kernel_width), stride=1, padding=0)
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
        
        if(wandb_config.dense==2):
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




        self.cov2fc_H = int( (traceLen[0]-wandb_config.kernel+1)/2 )
        #self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-wandb_config.kernel+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-wandb_config.kernel+1)/2 )
        #self.cov2fc_W = int( (self.cov2fc_W-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-wandb_config.kernel_width+1)/2 )


        # self.conv_2L = int( (self.conv_1L)/2 )
        
        self.feedback = None


        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=wandb_config.kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(wandb_config.kernel,wandb_config.kernel_width), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(wandb_config.kernel,wandb_config.kernel_width), stride=1, padding=0)
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
        
        if(wandb_config.dense==2):
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




        self.cov2fc_H = int( (traceLen[0]-wandb_config.kernel+1))
        #self.cov2fc_H = int( (self.cov2fc_H-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_H = int( (self.cov2fc_H-wandb_config.kernel+1)/2 )
        self.cov2fc_H = int( (self.cov2fc_H-wandb_config.kernel+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1)/4 )
        # self.conv_2L = int( (self.conv_1L)/2 )
        self.cov2fc_W = int( (traceLen[1]-wandb_config.kernel+1) )
        #self.cov2fc_W = int( (self.cov2fc_W-3+1)/2 )
        # self.cov2fc = int( (self.cov2fc-16+1) )
        self.cov2fc_W = int( (self.cov2fc_W-wandb_config.kernel_width+1)/2 )
        self.cov2fc_W = int( (self.cov2fc_W-wandb_config.kernel_width+1)/2 )


        # self.conv_2L = int( (self.conv_1L)/2 )
        
        self.feedback = None


        # UL paper non_profiing network
        self.conv1 = nn.Conv2d(self.conv1_inchannel, self.conv1_outchannel, kernel_size=wandb_config.kernel, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1_outchannel)
        # self.bn1 = BatchInstanceNorm1d(num_features=self.conv1_outchannel)
        self.conv2 = nn.Conv2d(self.conv2_inchannel, self.conv2_outchannel, kernel_size=(wandb_config.kernel,wandb_config.kernel_width), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2_outchannel)
        # self.bn2 = BatchInstanceNorm1d(num_features=self.conv2_outchannel)
        self.conv3 = nn.Conv2d(self.conv3_inchannel, self.conv3_outchannel, kernel_size=(wandb_config.kernel,wandb_config.kernel_width), stride=1, padding=0)
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
        

        if(wandb_config.dense==2):
            out1 = F.relu(self.fc2(out1))
            out1 = F.softmax(self.fc3(out1), dim=1)
        else:
            out1 = F.softmax(self.fc1(out1), dim=1)

        '''branch 2: bits compressing'''
        # out2 = F.selu(self.rfc1(x2))


        return out1
