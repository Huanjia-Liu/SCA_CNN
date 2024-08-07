import numpy as np
import torch
from math import comb
from sklearn.metrics import normalized_mutual_info_score
# import ddks
# from nn_label_transform import label_transform




def MSE( preds, labels ):
    loss = torch.mean((preds - labels)**2) 
    # print( loss.item() ) 
    return loss

'''function is for matrix use, here apply for array'''

def corr_loss( preds, labels ):
    p_models = labels.reshape( 1, labels.shape[0] ).float()
    trcs_m = preds.mean( dim=0 )
    pm_m = torch.reshape( p_models.mean(dim=1), (p_models.shape[0],1) )
    r_num = ( p_models - pm_m ).matmul( preds - trcs_m )
    r_den = torch.sqrt(torch.sum(( p_models - pm_m )**2,dim=1).reshape( (p_models.shape[0], 1) ).matmul( torch.sum(( preds - trcs_m )**2, dim=0).reshape( (1, preds.shape[1]) ) ))
    corr_mtx = r_num/r_den
    return (1-(corr_mtx[0,0]).abs())



def cross_entropy( preds, labels, bins=9 ):
    # preds = preds.squeeze( dim=1 )

    bins_l = torch.zeros( bins, dtype=torch.float ).to('cuda')
    bins_l_ele, bins_l_cnt = torch.unique( labels, sorted=True, return_counts=True, dim=0  )
    for idx in range(bins_l_ele.shape[0]):
        bins_l[bins_l_ele[idx].item()]=bins_l_cnt[idx].to('cuda').item()
    # bins_l = torch.tensor( [ bins_l[bins_l_ele[idx]]=bins_l_cnt[idx] for idx in bins_l_ele.shape[0] ] )
    prob_l = bins_l / labels.shape[0]
    # bins_p2 = torch.histc( preds, bins=9 )
    max_p = torch.max( preds )
    min_p = torch.min( preds )
    step_size = (max_p - min_p)/bins
    bins_p_ele = torch.zeros( bins+1, dtype=torch.float ).to('cuda')
    bins_p_ele[0] = min_p 
    bins_p_ele[-1] = max_p + 0.001
    for step in range( bins-1 ):
        bins_p_ele[step+1] = min_p + (step+1)*step_size
    bins_p = torch.zeros( bins, dtype=torch.float ).to('cuda')
    for idx in range( bins ):
        bins_p[idx] = torch.logical_and(bins_p_ele[idx]<=preds, preds<bins_p_ele[idx+1] ).sum()
        
    prob_p = bins_p / preds.shape[0]
    
    ce = torch.mean(-torch.sum(prob_l * torch.log(prob_p), dim=0))
    return ce

# @staticmethod
# def n_dim_KS_test( preds, labels ):
#     labels = labels.unsqueeze(dim=1)
#     calculation = ddks.methods.vdKS()
#     distance = calculation(preds.cpu(), labels.cpu())
#     return distance

'''KNLL from paper Imbalanced Data Problems in Deep Learning-Based Side-Channel Attacks: Analysis and Solution'''

def KNLL( preds, labels ):
    '''preds should be softmax output'''
    max_label = len(preds[0]) - 1
    invcoef = torch.zeros( max_label+1, device=preds.device )
    for i in range(max_label+1):
        invcoef[i] = 1/comb( max_label, i )
    # invcoef = torch.from_numpy( np.array( invcoef ) ) 

    #a = preds.gather( dim=1, index=labels.view(-1,1)).squeeze(1)
    #b = -torch.log(a)
    #c  = len(labels)
    #d = invcoef[labels]
    #KNLL =( b + d).sum()/c

    KNLL = ( -torch.log( preds.gather( dim=1, index=labels.view(-1,1)).squeeze(1) ) + invcoef[ labels ] ).sum()/len(labels)
    return KNLL


###########################################################################################
# Mutual inforamtion loss 

#calculate gussian kernel for pdf
def kernel_generate(values, smooth, unique):

    kde_label = unique
    label_len = kde_label.size()[0]


    delta_x = values.repeat(label_len,1).T - kde_label
    kernel_values = torch.exp(-0.5*(delta_x/smooth).pow(2))
    return kernel_values

#convert the kernels to pdf 
def pdf_generate(kernel):
    pdf = torch.mean(kernel,dim=0)
    normalization = torch.sum(pdf) + 1e-10
    pdf = pdf/normalization
    return pdf


#calculate guassian kernel for joint pdf
def joint_kernel_generate(kernel1, kernel2):
    
    joint_kernel = torch.matmul(kernel1.transpose(0,1), kernel2)
    return joint_kernel

#convert the joint kernel to joint pdf 
def joint_pdf_generate(kernel):
    normalization = torch.sum(kernel, dim=(0,1)) + 1e-10
    pdf = kernel / normalization
    return pdf




#MI output, pred and label are in torch format
def MI_output(pred_r,label, smooth, batch = 10000):

    unique_pred = torch.unique(pred_r) 
    unique_label = torch.unique(label)
    pred = pred_r.reshape(1,pred_r.shape[0])
    for i in range(pred.shape[0]//batch + 0 if (not pred.shape[0]%batch) else 1):
        batch_pred = pred[batch*i: batch*(i+1)]
        batch_label = label[batch*i: batch*(i+1)]
        
        if(i==0):
            kernel_pred = kernel_generate(batch_pred, smooth, unique_pred)
            kernel_label = kernel_generate(batch_label, smooth, unique_label)
            joint_kernel = joint_kernel_generate(kernel_pred[:, batch*i: batch*(i+1)], kernel_label[:, batch*i: batch*(i+1)])
        
        else:
            kernel_pred = torch.vstack((kernel_pred, kernel_generate(batch_pred, smooth)))
            kernel_label = torch.vstack((kernel_label, kernel_generate(batch_label, smooth)))
            joint_kernel = torch.add(joint_kernel, joint_kernel_generate(kernel_pred[batch*i: batch*(i+1),:], kernel_label[batch*i: batch*(i+1),:]))
        
    pdf_pred = pdf_generate(kernel_pred)
    pdf_label = pdf_generate(kernel_label)
    pdf_joint = joint_pdf_generate(joint_kernel)

    H_x1 = -torch.sum(pdf_pred*torch.log2(pdf_pred+1e-10))
    H_x2 = -torch.sum(pdf_label*torch.log2(pdf_label+1e-10))
    H_x1x2 = -torch.sum(pdf_joint*torch.log2(pdf_joint+1e-10), dim=(0,1))
    mutual_information = H_x1 + H_x2 - H_x1x2

    
    mutual_information = 2*mutual_information/(H_x1+H_x2)
    return(1-mutual_information)


######################################################################################











# def vcorrcoef(trcs, p_models):
#     trcs_m = np.mean(trcs, axis=0)
#     pm_m = np.reshape( np.mean(p_models, axis=1), (p_models.shape[0],1) )

#     r_num = ( p_models - pm_m ).dot( trcs - trcs_m )
#     r_den = np.sqrt(np.sum(( p_models - pm_m )**2,axis=1).reshape( (p_models.shape[0], 1) ).dot( np.sum(( trcs - trcs_m )**2, axis=0).reshape( (1, trcs.shape[1]) ) ))
#     corr_mtx = r_num/r_den
#     return corr_mtx


# def main():
#     # labels = np.zeros( 10 )
#     # preds = torch.as_tensor( labels, device=torch.device('cuda') )  
#     # index = [0,2,4,6,8]
#     # labels[ index ] = 1
#     # labels2 = torch.from_numpy( labels ).to( 'cuda' )

#     # preds = label_transform.make_one_hot( preds )
#     # labels2 = label_transform.make_one_hot( labels2 )

#     # loss_functions.MSE( preds, labels2 )

#     labels = torch.tensor( [5.,4.,3.,2.,1.] ).to('cuda')
#     preds = torch.tensor( [1.,2.,3.,4.,5.] ).reshape(5,1).to('cuda')
#     a = loss_functions.corr_loss( preds, labels )[0,0]
#     print()




# import time


# if "__main__" == __name__:
#     start_time = time.time()

#     main()

#     stop_time = time.time()

#     print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))
