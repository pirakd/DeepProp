import torch
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from torch import Tensor
import copy

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
        if intermediate_loss_weight:
            self.intermediate_loss_weight = torch.tensor(intermediate_loss_weight, requires_grad=False).to(self.device)
        else:
            self.intermediate_loss_weight = None


    def train(self, train_loader, eval_loader, model, max_evals_no_improvement=8):
        best_model = None
        self.losses = {'train': [], 'validation': []}
        self.metrics = {'train': [], 'validation': []}
        best_loss, best_auc, best_acc, best_epoch = 0,0,0,0
        best_eval_loss = np.inf
        no_improvement_counter = 0

        n_batches_per_epoch = int(np.ceil(len(train_loader.dataset) / train_loader.batch_size))
        n_samples_per_epoch = len(train_loader.dataset)

        for epoch in range(self.n_epochs):
            epoch_loss, epoch_intermediate_loss, epoch_classifier_loss, epoch_hits = self.train_single_epoch(model, train_loader)
            self.losses['train'].append(epoch_loss/(n_batches_per_epoch))
            self.metrics['train'].append(epoch_hits/n_samples_per_epoch)
            print('epoch {}, train_loss: {:.2e}, classifier_loss: {:.2e}, intermediate_loss:{:.2e} train_acc:{:.2f}'
                  .format(epoch, self.losses['train'][-1], epoch_classifier_loss/n_batches_per_epoch,
                          epoch_intermediate_loss/n_batches_per_epoch,  epoch_hits/n_samples_per_epoch))

            if (np.mod(epoch, self.eval_interval) == 0 and epoch) or (epoch+1 == self.n_epochs):
                avg_eval_loss, avg_eval_intermediate_loss, avg_eval_classifier_loss, eval_acc, eval_auc, _, _ = self.eval(model, eval_loader)

                self.losses['validation'].append(avg_eval_loss)
                self.metrics['validation'].append(eval_acc)
                print('epoch {}, validation loss:{:.2e}, validation classifier loss: {:.2e}, validation intermediate loss: {:.2e}, validation acc: {:.2f}, validation auc: {:.2f}'
                      .format(epoch, avg_eval_loss, avg_eval_classifier_loss, avg_eval_intermediate_loss,
                              eval_acc, eval_auc))

                if best_auc < eval_auc:
                    best_eval_loss = avg_eval_loss
                    no_improvement_counter = 0
                    best_loss, best_auc, best_acc, best_epoch, _, _ = avg_eval_loss, eval_auc, eval_acc, epoch, _, _
                    best_model = copy.deepcopy(model)
                else:
                    no_improvement_counter += 1

                if no_improvement_counter == max_evals_no_improvement:
                    print('early stopping on epoch {}'.format(epoch))
                    break

        train_stats = {'best_val_loss':best_eval_loss, 'best_acc':best_acc, 'best_auc':best_auc,
                       'best_epoch':best_epoch}
        return train_stats, best_model

    def train_single_epoch(self, model, train_loader):
        epoch_loss = 0
        epoch_intermediate_loss = 0
        epoch_classifier_loss = 0
        hits = 0
        for source_batch, terminal_batch, labels_batch in train_loader:
            self.optimizer.zero_grad()
            if self.device != 'cpu':
                source_batch, terminal_batch, labels_batch =\
                    (x.to(self.device) for x in[source_batch, terminal_batch, labels_batch])
            if torch.get_default_dtype() is torch.float32:
                source_batch, terminal_batch, = source_batch.float(), terminal_batch.float()

            out, pred, pre_pred = model(source_batch, terminal_batch)

            classifier_loss = self.criteria(out, labels_batch)
            if self.intermediate_loss_weight:
                intermediate_loss = self.intermediate_criteria(torch.squeeze(pre_pred),
                                        torch.repeat_interleave(labels_batch, model.n_experiments).to(torch.get_default_dtype()))
                epoch_intermediate_loss += intermediate_loss.item()

                loss = ((1-self.intermediate_loss_weight) * classifier_loss) +\
                       (self.intermediate_loss_weight * intermediate_loss)
            else:
                loss = classifier_loss

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_classifier_loss += classifier_loss.item()
            hits += torch.sum(pred == labels_batch).item()

        return epoch_loss, epoch_intermediate_loss, epoch_classifier_loss, hits

    def eval(self, model, eval_loader, output_probs= False):
        n_test_batches_per_epoch = int(np.ceil(len(eval_loader.dataset) / eval_loader.batch_size))
        eval_loss = 0
        epoch_intermediate_eval_loss = 0
        epoch_classifier_loss = 0
        hits = 0

        all_outs = []
        all_labels = []
        for source_batch, terminal_batch, labels_batch in eval_loader:
            if self.device != 'cpu':
                source_batch, terminal_batch, labels_batch =\
                    (x.to(self.device) for x in[source_batch, terminal_batch, labels_batch])
            if torch.get_default_dtype() is torch.float32:
                source_batch, terminal_batch, = source_batch.float(), terminal_batch.float()

            out, pred, pre_pred = model(source_batch, terminal_batch)

            classifier_loss = self.criteria(out, labels_batch)
            if self.intermediate_loss_weight:
                intermediate_loss = self.intermediate_criteria(torch.squeeze(pre_pred),
                                        torch.repeat_interleave(labels_batch, model.n_experiments).to(torch.get_default_dtype()))
                epoch_intermediate_eval_loss += intermediate_loss.item()

                loss = ((1-self.intermediate_loss_weight) * classifier_loss) +\
                       (self.intermediate_loss_weight * intermediate_loss)
            else:
                loss = classifier_loss

            eval_loss += loss.item()
            epoch_classifier_loss += classifier_loss.item()
            hits += torch.sum(pred == labels_batch).item()
            all_outs.append(out.cpu().detach().numpy())
            all_labels.append(labels_batch)

        # probs = torch.nn.functional.softmax(torch.squeeze(torch.cat(all_outs, 0)), dim=1).cpu().detach().numpy()
        probs = torch.nn.functional.softmax(torch.squeeze(Tensor(np.concatenate(all_outs, 0))), dim=1)
        all_labels = torch.squeeze(torch.cat(all_labels)).cpu().detach().numpy()

        if output_probs:
            return probs, all_labels
        else:
            precision, recall, thresholds = precision_recall_curve(all_labels, probs[:, 1])
            mean_auc = auc(recall, precision)

            avg_eval_loss = eval_loss / n_test_batches_per_epoch
            avg_eval_intermediate_loss = epoch_intermediate_eval_loss / n_test_batches_per_epoch
            avg_eval_classifier_loss = epoch_classifier_loss / n_test_batches_per_epoch
            eval_acc = hits / len(eval_loader.dataset)
            return avg_eval_loss, avg_eval_intermediate_loss, avg_eval_classifier_loss, eval_acc, mean_auc, precision, recall
