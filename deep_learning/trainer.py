import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from torch import Tensor

class Trainer():
    def __init__(self, n_epochs, criteria, optimizer, eval_metric,  eval_interval=1,  device='cpu', verbose=3, early_stop=None):
        self.n_epochs = n_epochs
        self.criteria = criteria
        self.optimizer = optimizer
        self.eval_metric = eval_metric
        self.device = device
        self.eval_interval = eval_interval


    def train(self, train_loader, eval_loader, model):
        raise NotImplemented


    def train_single_epoch(self, model, train_loader):
        raise NotImplemented

    def eval(self, model, eval_loader):
        raise NotImplemented

    def test(self, model, test_loader, metric=None):
        raise NotImplemented


class ClassifierTrainer(Trainer):
    def __init__(self, n_epochs, criteria, intermediate_criteria, intermediate_loss_weight,  optimizer,
                 eval_metric, eval_interval, device, verbose=3, early_stop=None):
        super().__init__(n_epochs, criteria, optimizer, eval_metric, eval_interval, device, verbose, early_stop)
        self.intermediate_criteria = intermediate_criteria
        self.intermediate_loss_weight = torch.tensor(intermediate_loss_weight, requires_grad=False).to(self.device)


    def train(self, train_loader, eval_loader, model, max_evals_no_improvement=8):
        self.losses = {'train': [], 'validation': []}
        self.metrics = {'train': [], 'validation': []}
        best_loss, best_auc, best_acc, best_epoch, best_precision, best_recall = 0,0,0,0,0,0
        best_eval_loss = np.inf
        no_improvement_counter = 0

        n_batches_per_epoch = int(np.ceil(len(train_loader.dataset) / train_loader.batch_size))
        n_samples_per_epoch = len(train_loader.dataset)

        n_test_batches_per_epoch = int(np.ceil(len(eval_loader.dataset) / eval_loader.batch_size))
        for epoch in range(self.n_epochs):
            epoch_loss, epoch_intermediate_loss, epoch_hits = self.train_single_epoch(model, train_loader)
            self.losses['train'].append(epoch_loss/(n_batches_per_epoch))
            self.metrics['train'].append(epoch_hits/n_samples_per_epoch)
            print('epoch {}, train_loss: {:.2e}, intermediate_loss:{:.2e} train_acc:{:.2f}'
                  .format(epoch, self.losses['train'][-1], epoch_intermediate_loss/n_batches_per_epoch,  epoch_hits/n_samples_per_epoch))
            if np.mod(epoch, self.eval_interval) == 0 and epoch:
                eval_loss, eval_intermediate_loss, eval_hits, eval_auc, precision, recall = self.eval(model, eval_loader)
                avg_eval_loss = eval_loss / n_test_batches_per_epoch
                eval_acc = eval_hits/len(eval_loader.dataset)
                self.losses['validation'].append(avg_eval_loss)
                self.metrics['validation'].append(eval_acc)
                print('epoch {},test_loss:{:.2e}, test_acc: {:.2f}, test_auc: {:.2f}'.format(epoch, avg_eval_loss, eval_acc, eval_auc))

                if avg_eval_loss < best_eval_loss:
                    best_eval_loss =avg_eval_loss
                    no_improvement_counter = 0
                    best_loss, best_auc, best_acc, best_epoch, best_precision, best_recall = avg_eval_loss, eval_auc, eval_acc, epoch, precision, recall

                else:
                    no_improvement_counter += 1

                if no_improvement_counter == max_evals_no_improvement:
                    print('early stopping on epoch {}'.format(epoch))
                    return best_eval_loss, best_acc, best_auc, best_epoch, best_precision, best_recall

        return best_eval_loss, best_acc, best_auc, best_epoch, best_precision, best_recall

    def train_single_epoch(self, model, train_loader):
        epoch_loss = 0
        intermediate_loss = 0
        hits = 0

        for source_batch, terminal_batch, labels_batch in train_loader:
            self.optimizer.zero_grad()
            if self.device != 'cpu':
                source_batch, terminal_batch, labels_batch =\
                    (x.to(self.device) for x in[source_batch, terminal_batch, labels_batch])
            out, pred, pre_pred = model(source_batch, terminal_batch)
            intermediate_loss = self.intermediate_criteria(torch.squeeze(torch.sigmoid(pre_pred)),
                                    torch.repeat_interleave(labels_batch, model.n_experiments).to(torch.float64))

            loss = ((1-self.intermediate_loss_weight) * self.criteria(out, labels_batch)) +\
                   (self.intermediate_loss_weight * intermediate_loss)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            intermediate_loss += intermediate_loss
            hits += torch.sum(pred == labels_batch).item()

        return epoch_loss, intermediate_loss, hits


    def eval(self, model, eval_loader):
        eval_loss = 0
        intermediate_eval_loss = 0
        hits = 0

        all_outs = []
        all_labels = []
        for source_batch, terminal_batch, labels_batch in eval_loader:
            if self.device != 'cpu':
                source_batch, terminal_batch, labels_batch =\
                    (x.to(self.device) for x in[source_batch, terminal_batch, labels_batch])
            out, pred, pre_pred = model(source_batch, terminal_batch)
            intermediate_loss = self.intermediate_criteria(torch.squeeze(torch.sigmoid(pre_pred)),
                                    torch.repeat_interleave(labels_batch, model.n_experiments).to(torch.float64))

            loss = (1-self.intermediate_loss_weight) * self.criteria(out, labels_batch) +\
                   (self.intermediate_loss_weight * intermediate_loss)
            eval_loss += loss.item()
            intermediate_eval_loss += intermediate_loss
            hits += torch.sum(pred == labels_batch).item()
            all_outs.append(out)
            all_labels.append(labels_batch)

        probs = torch.nn.functional.softmax(torch.squeeze(torch.cat(all_outs, 0)), dim=1).detach().numpy()
        all_labels = torch.squeeze(torch.cat(all_labels)).cpu().detach().numpy()
        precision, recall, thresholds = precision_recall_curve(all_labels, probs[:, 1])
        mean_auc = auc(recall, precision)
        return eval_loss, intermediate_eval_loss, hits, mean_auc, precision, recall