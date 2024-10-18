import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch_geometric
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F

from embedder import embedder
from src.utils import cluster_acc
from layers import GNN_Encoder, MeanAct, DispAct, ZINBLoss

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score

from src.transform import Hete_DropFeatures
from torch import optim


class scZGCL_Trainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self.args = args
        self.old_celltype_result = []
        
    def _init_model(self):
        layers = eval(self.args.layers)
        heads=eval(self.args.heads)

        self.model = scZGCL(self.adata.n_vars, self.args.n_clusters,self.args.dropout, layers,heads, 
                            self.args.tau, self.args.alpha, self.device).to(self.device)

        param_group = [{'params' : self.model.parameters(), 'lr':self.args.lr, 'weight_decay':self.args.decay}]
        self.optimizer = optim.AdamW(param_group)        
        self.recon_loss = ZINBLoss().to(self.args.device)

        self.transform1 = Hete_DropFeatures(self.args.df_1)
        self.transform2 = Hete_DropFeatures(self.args.df_2)

    def pre_train(self):
        for epoch in range(1, self.args.epochs+1):
            total_examples = total_recon_loss = total_con_loss = 0
            # establish two views
            for view1, view2 in zip(self.train_loader1, self.train_loader2):
                self.model.train()
                self.optimizer.zero_grad()

                batch_size = view1.batch_size
                sampled_id = view1.n_id[:batch_size]

                view1 = self.transform1(view1).to(self.args.device)
                view2 = self.transform2(view2).to(self.args.device)            
                
                mean1, disp1, pi1, rep1 = self.model(view1)
                mean1 = mean1[:batch_size]
                disp1 = disp1[:batch_size]
                pi1 = pi1[:batch_size]
                rep1 = rep1[:batch_size]

                mean2, disp2, pi2, rep2 = self.model(view2)
                mean2 = mean2[:batch_size]
                disp2 = disp2[:batch_size]
                pi2 = pi2[:batch_size]
                rep2 = rep2[:batch_size]

                sf = torch.tensor(self.adata.obs.size_factors)[sampled_id].to(self.args.device)
                X = torch.tensor(self.adata.raw.X)[sampled_id].to(self.args.device)

                recon_loss = self.recon_loss(X, mean1, disp1, pi1, sf)
                recon_loss += self.recon_loss(X, mean2, disp2, pi2, sf)
                recon_loss /= 2

                ######################### Contrastive Loss ####################
                con_loss = self.model.ins_contrastive_loss(rep1, rep2, device=self.args.device)
                ##############################################################

                # calculate loss
                loss = recon_loss + self.args.lam1 * con_loss              
                
                loss.backward()
                self.optimizer.step()

                total_examples += batch_size
                total_recon_loss += float(recon_loss) * batch_size
                total_con_loss += float(con_loss) * batch_size

            total_recon_loss /= total_examples
            total_con_loss /= total_examples

            st = '** epochs:{}/{} | recon : {:.4f} | con-wise : {:.4f}  **'.format(
                    epoch,self.args.epochs, total_recon_loss, total_con_loss)

            self.logger.info(st)

            if epoch > (self.args.epochs // 2):
                flag = self.Pretrain_Evaluate_Convergence(epoch)
                if flag == 1:
                    break


    def fine_tune(self):
        param_group = [ {'params' : self.model.parameters(), 'lr':self.args.fine_lr, 'weight_decay':self.args.decay, 'rho':.95}]
        self.optimizer = optim.Adadelta(param_group)

        self.logger.info("Initializing cluster centers with kmeans.")
        # perform K-Means clustering
        kmeans = KMeans(self.model.n_clusters, n_init=20)
        cell_rep = self.model.predict_full_cell_rep(self.eval_loader)
        self.y_pred = kmeans.fit_predict(cell_rep)
        self.y_pred_last = self.y_pred
        self.model.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32))
        
        for epoch in range(1, self.args.fine_epochs+1):
            self.model.eval()
            latent = self.model.predict_full_cell_rep(self.eval_loader)
                          
            q = self.model.soft_assign(torch.tensor(latent).to(self.args.device))
            p = self.model.target_distribution(q).data

            self.y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            ari = adjusted_rand_score(self.adata.obs['Group'], self.y_pred)
            nmi = normalized_mutual_info_score(self.adata.obs['Group'], self.y_pred)
            ca, ma_f1, mi_f1 = cluster_acc(self.adata.obs['Group'], self.y_pred)
            

            ########################## Evaluate Convergence ############################
            num = self.adata.X.shape[0]
            delta_label = np.sum(self.y_pred != self.y_pred_last).astype(np.float32) / num
            self.y_pred_last = self.y_pred
            
            # Record latent
            if epoch == self.args.fine_epochs or (epoch > 1 and delta_label < self.args.tol):
                with open(self.args.name + '_zinb_latent.txt', 'a') as f_2:
                    np.savetxt(f_2, latent, fmt='%d', delimiter=' ', newline='\n')
            
            # Record Final Clustering Result
            if epoch == self.args.fine_epochs or (epoch > 1 and delta_label < self.args.tol):
                with open(self.args.name + '_zinb_label.txt', 'a') as f:
                    np.savetxt(f, self.y_pred, fmt='%d', delimiter=' ', newline='\n')
            
            if epoch > 1 and delta_label < self.args.tol:
                self.logger.info(f'delta_label {delta_label} < tol {self.args.tol}')
                self.logger.info("Reach tolerance threshold. Stopping fine-tuning.")
                break
            
            total_examples = total_recon_loss = total_cluster_loss = 0
            for batch in self.train_loader1:
                self.model.train()
                batch_size = batch.batch_size
                sampled_id = batch.n_id[:batch_size]
                
                batch = self.transform1(batch).to(self.args.device)
                
                mean, disp, pi, rep = self.model(batch)
                mean = mean[:batch_size]
                disp = disp[:batch_size]
                pi = pi[:batch_size]
                rep = rep[:batch_size]

                ########################## Cluster Loss ######################
                qbatch = self.model.soft_assign(rep)
                pbatch = p[sampled_id]
                target = Variable(pbatch).to(self.args.device)

                cluster_loss = self.model.cluster_loss(target, qbatch)
                ##############################################################

                sf = torch.tensor(self.adata.obs.size_factors)[sampled_id].to(self.args.device)
                X = torch.tensor(self.adata.raw.X)[sampled_id].to(self.args.device)
                recon_loss = self.recon_loss(X, mean, disp, pi, sf)

                loss = recon_loss + self.args.lam2 * cluster_loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_examples += batch_size
                total_recon_loss += float(recon_loss) * batch_size
                total_cluster_loss += float(cluster_loss) * batch_size

            total_recon_loss /= total_examples
            total_cluster_loss /= total_examples            

            st = '** epochs:{}/{} | recon : {:.4f} | cluster : {:.4f} | nmi : {:.4F} | ca : {:.4F} | ari : {:.2f} | ma-f1 : {:.4F} | mi-f1 : {:.4F} | **'.format(
                epoch, self.args.fine_epochs, total_recon_loss, total_cluster_loss, nmi, ca, ari, ma_f1, mi_f1)

            self.logger.info(st)

        return nmi, ca, ari, ma_f1, mi_f1

    def train(self):

        self._init_dataset(self.args.seed) #
        self._init_model()  
        self.pre_train()
        nmi, ca, ari, ma_f1, mi_f1 = self.fine_tune()
        
        if self.args.save_model:
            torch.save({'model_state_dict' : self.model.state_dict()}, self.model_path)

        self.logger.info(f'================================ Final Score =======================================')
        self.logger.info("** NMI : {:.4F} |  CA : {:.4F} | ARI : {:.4f} | Ma-F1 : {:.4f} | Mi-F1 : {:.4f} | **\n".format(nmi, ca, ari, ma_f1, mi_f1))
        self.logger.info(f'=========================================================================================')
        

        

class scZGCL(nn.Module):
    def __init__(self,num_gene, n_clusters, dropout,layer_sizes=[256,64], heads=[4,1],tau=0.25, alpha=1.0, device=0):
        super(scZGCL, self).__init__()

        self.tau = tau
        self.alpha = alpha
        self.emb_dim = layer_sizes[-1]
        
        self.encoder = GNN_Encoder(num_gene, dropout, layer_sizes,heads)
        decoder_sizes = layer_sizes[::-1]
        layers = []
        for in_dim, out_dim in zip(decoder_sizes[:-1], decoder_sizes[1:]):
            layers.append((nn.Linear(in_dim, out_dim)))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

        hidden = decoder_sizes[-1]
        self.mean_layer = nn.Sequential(nn.Linear(hidden, num_gene), MeanAct())
        self.disp_layer = nn.Sequential(nn.Linear(hidden, num_gene), DispAct())
        self.pi_layer = nn.Sequential(nn.Linear(hidden, num_gene), nn.Sigmoid())
        self.softmax = nn.Softmax(dim=1)

        self.mu = nn.Parameter(torch.Tensor(n_clusters, self.emb_dim))

        self.n_clusters = n_clusters
        self.device = device

        self.reset_parameters()

    def encode(self, x):
        rep = self.encoder(x)
        return rep

    def decode(self, z):

        hidden = self.decoder(z)

        mean = self.mean_layer(hidden)
        disp = self.disp_layer(hidden)
        pi = self.pi_layer(hidden)

        return mean, disp, pi

    def forward(self, view):
        
        rep = self.encode(view)
        mean, disp, pi = self.decode(rep)

        return mean, disp, pi, rep

    def ins_contrastive_loss(self, rep1, rep2, device=None):

        batch_size = rep1.size(0)

        pos_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        neg_mask = 1 - pos_mask
        
        rep1 = F.normalize(rep1, dim=1)
        rep2 = F.normalize(rep2, dim=1)
        contrast_feature = torch.concat((rep1, rep2), dim=0)
        anchor_feature = contrast_feature    
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, anchor_feature.T),
            self.tau)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        pos_mask = pos_mask.repeat(2, 2)
        neg_mask = neg_mask.repeat(2, 2)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        pos_mask = pos_mask * logits_mask


        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        exp_logits = exp_logits * (neg_mask+pos_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(2, batch_size).mean()

        return loss
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**((self.alpha+1.0)/2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q
    
    def target_distribution(self, q):
        p = q**2 / q.sum(0)
        return (p.t() / p.sum(1)).t()

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target*torch.log(target/(pred+1e-6)), dim=-1))
        kldloss = kld(p, q)
        return kldloss

    def infer_num_proto(self):
        num_proto = eval(self.granularity)
        if len(num_proto) == 1:
            self.n_cluster_list = [self.n_clusters * num_proto[0]]

        elif len(num_proto) == 2:
            cluster1 = self.n_clusters * num_proto[0]
            cluster2 = int(max(cluster1+1, np.ceil(self.n_clusters * num_proto[1])))
            self.n_cluster_list = [cluster1, cluster2]

        elif len(num_proto) == 3:
            cluster1 = int(np.floor(self.n_clusters * num_proto[0]))
            cluster2 = int(max(cluster1+1, self.n_clusters * num_proto[1]))
            cluster3 = int(max(cluster2+1, np.ceil(self.n_clusters * num_proto[2])))
            self.n_cluster_list = [cluster1, cluster2, cluster3]

    def predict_celltype(self, rep, n_clusters=None):

        if n_clusters is None:
            n_clusters = self.n_clusters

        kmeans = KMeans(n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(rep)

        return y_pred

    def predict_full_cell_rep(self, loader):
        self.eval()
        cell_reps = []
        for batch in loader:
            batch_size = batch.batch_size
            batch = batch.to(self.device)
            cell_rep = self.encoder(batch)
            cell_rep = cell_rep[:batch_size].detach().to('cpu').numpy()
            cell_reps.append(cell_rep)

        cell_rep = np.concatenate(cell_reps)
        return cell_rep

    def init_mu(self):
        self.mu = nn.Parameter(torch.Tensor(self.n_clusters, self.emb_dim).to(self.device))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
            elif isinstance(m, torch_geometric.nn.Linear):
                m.reset_parameters()