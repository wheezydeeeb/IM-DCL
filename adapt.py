import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.nn.functional as F
import os
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import sqrtm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import configs

from io_utils import model_dict, parse_args

from datasets import ISIC_few_shot_base, EuroSAT_few_shot_base, CropDisease_few_shot_base, Chest_few_shot_base


class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x
        
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
    
   
def sup(idx, dis):
    numbers = [0, 16, 32, 48, 65]
    
    mask = torch.zeros_like(idx, dtype=torch.bool)
    for num in numbers:
        mask |= (idx == num)
    dis[mask] *= 2
    return dis

'''        
def tsne(k, name, epoch, features, labels):
    features = np.array(features.cpu())
    labels = np.array(labels.cpu())
    tsne = TSNE(n_components=2, random_state=42)
    embedded_features = tsne.fit_transform(features)
    plt.figure(figsize=(5, 3), dpi=600)
    plt.axis('off')
    colors = ['red', 'green', 'blue', 'black', 'orange']
    for i in range(len(labels)):
        plt.scatter(embedded_features[i, 0], embedded_features[i, 1], c=colors[labels[i]], s=5, cmap=None)
    if epoch % 100 == 0 or epoch == 299:
        save_path = '/scratch/project_2002243/huali/sourcefree/FTEM_BSR_CDFSL/tsne/' + str(name) + "_" + str(k) + "_" + str(epoch) + '.png'
        plt.savefig(save_path)
        plt.close()
'''


def finetune(name, novel_loader, n_query=15, freeze_backbone=False, n_way=5, n_support=5):

    iter_num = len(novel_loader) 
    #iter_test = iter(novel_loader)

    acc_all_ori = []
    acc_all_lp = []

    if params.use_saved:
        save_dir = '%s/checkpoints' % configs.save_dir
    else:
        save_dir = '%s/checkpoints' % configs.save_dir

    #k = 0
    for _, (x, y) in enumerate(novel_loader):
        ###############################################################################################
        # load pretrained model on miniImageNet
        #assert k <= 10
        pretrained_model = model_dict[params.model]()

        checkpoint_dir = '%s/%s_%s' %(save_dir, params.model, params.method)
        if params.train_aug:
            checkpoint_dir += '_aug'
        modelfile = os.path.join(checkpoint_dir, '%s_%s_1200.tar' % (params.model, params.method))
        tmp = torch.load(modelfile)
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.","")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'  
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        pretrained_model.load_state_dict(state)
        ###############################################################################################

        classifier = Classifier(512, n_way)

        ###############################################################################################
        n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)
    
        batch_size = 5
        support_size = n_way * n_support
        all_size = n_way * (n_support + n_query)
        s_z = support_size

        y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda()
        y_var = Variable(torch.from_numpy(np.repeat(range(n_way), (n_support+n_query)))).cuda()

        x_b_i = x_var[:, n_support:, :, :, :].contiguous().view(n_way * n_query, *x.size()[2:])
        x_a_i = x_var[:, :n_support, :, :, :].contiguous().view(n_way * n_support, *x.size()[2:])
        x_var_i = x_var[:, :, :, :, :].contiguous().view(n_way * (n_support + n_query), *x.size()[2:])
        x_a_i.cuda()
        x_b_i.cuda()
        y_i = y_a_i
        
        fea_bank = torch.randn(s_z, 512)
        score_bank = torch.randn(s_z, n_way).cuda()

        ###############################################################################################
        loss_fn = nn.CrossEntropyLoss().cuda()
        classifier_opt = torch.optim.SGD(classifier.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)

        if freeze_backbone is False:
            delta_opt = torch.optim.SGD(filter(lambda p: p.requires_grad, pretrained_model.parameters()), lr = 0.01)

        pretrained_model.cuda()
        classifier.cuda()
        ###############################################################################################
        total_epoch = 300

        if freeze_backbone is False:
            pretrained_model.train()
        else:
            pretrained_model.eval()
        
        classifier.train()

        for epoch in range(total_epoch):
            y_i.cuda()
            rand_id = np.random.permutation(support_size)
            alpha = (1 + 10 * epoch / total_epoch)**(-5.0) * 1.0
            
            '''
            with torch.no_grad():
                output_var = pretrained_model(x_var_i)
                output_var = output_var.view(output_var.size(0), -1)
                output = classifier(output_var)
            tsne(k, name, epoch, output, y_var)
            '''
            
            
            for j in range(0, support_size, batch_size):
                classifier_opt.zero_grad()
                if freeze_backbone is False:
                    delta_opt.zero_grad()

                #####################################
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                z_batch = x_a_i[selected_id]
                y_batch = y_i[selected_id] 
                #####################################

                output_z = pretrained_model(z_batch)
                output_z = output_z.view(output_z.size(0), -1)
                output = classifier(output_z)
                clf_loss = loss_fn(output, y_batch)
                
                softmax_out = nn.Softmax(dim=1)(output)
                entropy_loss = torch.mean(Entropy(softmax_out))
                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy_loss
                im_loss = entropy_loss * 1.0
                
                #clf_loss = loss_fn(output, y_batch)
                #loss = clf_loss
                loss = clf_loss + im_loss

                #####################################
                loss.backward()

                classifier_opt.step()
                
                if freeze_backbone is False:
                    delta_opt.step()
                    
            #####################################
            classifier_opt.zero_grad()
            if freeze_backbone is False:
                delta_opt.zero_grad()
                
            #####################################
            output_var = pretrained_model(x_var_i)
            output_var = output_var.view(output_var.size(0), -1)
            output = classifier(output_var)
            
            
            softmax_out = nn.Softmax(dim=1)(output)
            
            entropy_loss = torch.mean(Entropy(softmax_out))
            msoftmax = softmax_out.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
            entropy_loss -= gentropy_loss
            im_loss = entropy_loss * 1.0
            
            
            with torch.no_grad():
                output_f_norm = F.normalize(output_var)
                output_f_ = output_f_norm.cpu().detach().clone()
                pred_bs = softmax_out
    
                fea_bank = output_f_.detach().cpu().clone()
                score_bank = softmax_out.detach().clone()
                
                distance = output_f_ @ fea_bank.T
                dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=n_way + 1)
                idx_near = idx_near[:, 1:]  #batch x K
                dis_near = dis_near[:, 1:]  #batch x K
                dis_near = sup(idx_near, dis_near)
                score_near = score_bank[idx_near]  #batch x K x C

            # nn
            softmax_out_un = softmax_out.unsqueeze(1).expand(-1, n_way, -1)  # batch x K x C
            norm_dis = F.normalize(dis_near, p=1, dim=1).cuda()
            score_near = torch.mul(score_near.cuda(), norm_dis.unsqueeze(-1).cuda())
            ad_loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))
            mask = torch.ones((x_var_i.shape[0], x_var_i.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy = softmax_out.T#.detach().clone()#
            dot_neg = softmax_out @ copy  # batch x batch
            dot_neg = (dot_neg * mask.cuda()).sum(-1)  #batch
            neg_pred = torch.mean(dot_neg)
            aad_loss = ad_loss + neg_pred * alpha
            
            
            loss = im_loss + 0.1 * aad_loss
            #####################################
            loss.backward()

            classifier_opt.step()
            if freeze_backbone is False:
                delta_opt.step()
            
        k = k + 1
            
        with torch.no_grad():
            output = pretrained_model(x_b_i)
            output = output.view(output.size(0), -1)
            scores = classifier(output)
            x_lp = output.cpu().numpy()
            y_lp = F.softmax(scores, 1).cpu().numpy()
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind_ori = topk_labels.cpu().numpy()
       
        y_query = np.repeat(range(n_way), n_query)

        neigh = NearestNeighbors(params.k_lp)
        neigh.fit(x_lp)
        d_lp, idx_lp = neigh.kneighbors(x_lp)
        d_lp = np.power(d_lp, 2)
        sigma2_lp = np.mean(d_lp)

        n_lp = len(y_query)
        del_n = int(n_lp * (1.0 - params.delta))
        for i in range(n_way):
            yi = y_lp[:, i]
            top_del_idx = np.argsort(yi)[0:del_n]
            y_lp[top_del_idx, i] = 0

        w_lp = np.zeros((n_lp, n_lp))
        for i in range(n_lp):
            for j in range(params.k_lp):
                xj = idx_lp[i, j]
                w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
        q_lp = np.diag(np.sum(w_lp, axis=1))
        q2_lp = sqrtm(q_lp)
        q2_lp = np.linalg.inv(q2_lp)
        L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
        a_lp = np.eye(n_lp) - params.alpha * L_lp
        a_lp = np.linalg.inv(a_lp)
        ynew_lp = np.matmul(a_lp, y_lp)

        count_this = len(y_query)

        top1_correct_ori = np.sum(topk_ind_ori[:, 0] == y_query)
        correct_ori = float(top1_correct_ori)
        print('BSR: %f' % (correct_ori / count_this * 100))
        acc_all_ori.append((correct_ori / count_this * 100))

        topk_ind_lp = np.argmax(ynew_lp, 1)
        top1_correct_lp = np.sum(topk_ind_lp == y_query)
        correct_lp = float(top1_correct_lp)
        print('BSR+LP: %f' % (correct_lp / count_this * 100))
        acc_all_lp.append((correct_lp / count_this * 100))
        ###############################################################################################

    acc_all_ori  = np.asarray(acc_all_ori)
    acc_mean_ori = np.mean(acc_all_ori)
    acc_std_ori  = np.std(acc_all_ori)
    print('BSR: %d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num,  acc_mean_ori, 1.96 * acc_std_ori / np.sqrt(iter_num)))

    acc_all_lp = np.asarray(acc_all_lp)
    acc_mean_lp = np.mean(acc_all_lp)
    acc_std_lp = np.std(acc_all_lp)
    print('BSR+LP: %d Test Acc = %4.2f%% +- %4.2f%%' %
          (iter_num, acc_mean_lp, 1.96 * acc_std_lp / np.sqrt(iter_num)))


if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('finetune')
    print(params.test_n_way)
    print(params.n_shot)

    image_size = 224
    iter_num = 600
    params.method = 'ce'

    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot, n_query=100)
    #params.freeze_backbone = True
    freeze_backbone = params.freeze_backbone

    if params.dtarget == 'ISIC':
        print ("Loading ISIC")
        datamgr = ISIC_few_shot_base.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
        name = "ISIC"
    elif params.dtarget == 'EuroSAT':
        print ("Loading EuroSAT")
        datamgr = EuroSAT_few_shot_base.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
        name = "EuroSAT"
    elif params.dtarget == 'CropDisease':
        print ("Loading CropDisease")
        datamgr = CropDisease_few_shot_base.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
        name = "CropDisease"
    elif params.dtarget == 'ChestX':
        print ("Loading ChestX")
        datamgr = Chest_few_shot_base.SetDataManager(image_size, n_eposide=iter_num, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
        name = "ChestX"

    print (params.dtarget)
    print (freeze_backbone)
    finetune(name, novel_loader, freeze_backbone=freeze_backbone, **few_shot_params)
