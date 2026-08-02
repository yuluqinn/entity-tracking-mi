"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import os
import logging

from collections import defaultdict

from tqdm import tqdm
import numpy as np
import json
import sklearn
import torch
from torch.utils.data.dataloader import DataLoader
from matplotlib import pyplot as plt

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)

MAX_NUM_OPERATIONS = 12


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    debug_train = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.scheduler = None
        self.optimizer = None
        self.num_class = self.model.probe_class

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        elif torch.backends.mps.is_available():
            self.device = f'mps'
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []
        # adding triv/non-triv acc
        self.train_acc_nontriv_cont = []
        self.test_acc_nontriv_cont = []
        self.train_acc_triv_cont = []
        self.test_acc_triv_cont = []
        # would be a list of T-long, each is a lits of MAX_NUM_OPERATIONS-long, for stratified accuracies        
        self.train_strat_acc_cont = []
        self.test_strat_acc_cont = []
        # other metrics calculated using masks
        self.train_acc_mask_cont = defaultdict(list)
        self.test_acc_mask_cont = defaultdict(list)

    def flush_plot(self, save_path=None):
        # plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
        axs = axs.flat
        axs[0].plot(self.train_loss_cont, label="train")
        axs[0].plot(self.test_loss_cont, label="test")
        axs[0].set_title("Loss")
        axs[0].legend()
        axs[1].plot(self.train_acc_cont, label="train")
        axs[1].plot(self.test_acc_cont, label="test")
        axs[1].set_title("Accuracy")
        axs[1].legend()
        if save_path is not None:
            fig.savefig(save_path)
        # plt.show()
        # return a figure object

    def save_traces(self, ):
        tbd = {
            "train_loss_cont": self.train_loss_cont, "test_loss_cont": self.test_loss_cont,
            "train_acc_cont": self.train_acc_cont, "test_acc_cont": self.test_acc_cont,
            "train_strat_acc_cont": self.train_strat_acc_cont, "test_strat_acc_cont": self.test_strat_acc_cont,
            "train_acc_nontriv_cont": self.train_acc_nontriv_cont, "test_acc_nontriv_cont": self.test_acc_nontriv_cont,
            "train_acc_triv_cont": self.train_acc_triv_cont, "test_acc_triv_cont": self.test_acc_triv_cont,
            "train_acc_mask_cont": self.train_acc_mask_cont, "test_acc_mask_cont": self.test_acc_mask_cont,
        }
        with open(os.path.join(self.config.ckpt_path, "tensorboard.txt"), "w") as f:
            f.write(json.dumps(tbd) + "\n")

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(raw_model.state_dict(), os.path.join(self.config.ckpt_path, "checkpoint.ckpt"))

    def load_checkpoint(self):
        torch.load(os.path.join(self.config.ckpt_path, "checkpoint.ckpt"), map_location=self.device)

    def predict_old(self, prt=True):
        model, config = self.model, self.config
        model.train(False)
        data = self.test_dataset
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        pbar = enumerate(loader)
        losses = []
        totals_epoch = np.zeros(MAX_NUM_OPERATIONS,
                                dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
        hits_epoch = np.zeros(MAX_NUM_OPERATIONS,
                              dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS,
                                      dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_triv_epoch = np.zeros(MAX_NUM_OPERATIONS,
                                      dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS

        totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS,
                                        dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        totals_epoch_triv = np.zeros(MAX_NUM_OPERATIONS,
                                        dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS

        predictions = []

        for _, (x, y, age, mentioned) in pbar:
            x = x.to(self.device)  # [B, f]
            y = y.to(self.device)  # [B, #task=64] 
            age = age.to(self.device)  # [B, #task=64], in 0--59
            mentioned = mentioned.to(self.device)  # [B, #task]
            with torch.set_grad_enabled(False):
                logits, loss = model(x, y)
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                totals_epoch_triv += np.array([torch.sum((age == i) * (1-mentioned)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)

                y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                hits = y_hat == y  # [B, #task]
                hits_nontrivial = (y_hat == y) * mentioned
                hits_trivial = (y_hat == y) * (1-mentioned)
                hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                hits_triv_epoch += np.array([torch.sum(hits_trivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                predictions.append(y_hat.cpu().numpy())

        test_loss = float(np.mean(losses))
        test_acc = np.sum(hits_epoch).item() / (np.sum(totals_epoch).item() + 1e-20)
        test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / (np.sum(totals_epoch_nontriv).item() + 1e-20)
        test_acc_triv = np.sum(hits_triv_epoch).item() / (np.sum(totals_epoch_triv).item() + 1e-20)

        if prt:
            logger.info(f"test loss {test_loss:.5f}; test acc {test_acc * 100:.2f}%;  test non-triv acc {test_acc_nontriv * 100:.2f}%;  test triv acc {test_acc_triv * 100:.2f}%")
        predictions_matrix = np.concatenate(predictions, axis=0)
        return predictions_matrix

    def run_epoch(self, split, prt=True, epoch=-1):
        model, config = self.model, self.config
        num_classes = model.module.probe_class if hasattr(model, "module") else model.probe_class
        is_train = split == 'train'
        raw_model = model.module if hasattr(self.model, "module") else model
        if is_train and self.optimizer is None:
            self.optimizer, self.scheduler = raw_model.configure_optimizers(config)

        model.train(is_train)
        data = self.train_dataset if is_train else self.test_dataset
        loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        losses = []
        totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
        hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_triv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS

        recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FN)
        precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FP)
        total_recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FN 
        total_precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FP
             


        if hasattr(data, 'mask_fields'):
            hits_masks_epoch = {k:np.zeros(MAX_NUM_OPERATIONS,dtype=float) for k in data.mask_fields}
            hits_masks_epoch["confusion_matrix"] = np.zeros((num_classes, num_classes), dtype=float)

        totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS,
                                        dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        totals_epoch_triv = np.zeros(MAX_NUM_OPERATIONS,
                                     dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS

        if hasattr(data, 'mask_fields'):
            totals_epoch_masks = {k: np.zeros(MAX_NUM_OPERATIONS, dtype=float) for k in data.mask_fields}

        predictions = []
        gt_ls = []
        mentioned_ls = []
        total_mentioned = 0
        results = [] # save predictions for further analysis, if needed
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not prt, leave=True) if is_train else enumerate(loader)
        for it, (x, y, age, mentioned) in pbar:
            x = x.to(self.device)  # [B, f]
            y = y.to(self.device)  # [B, #task=64]
            age = age.to(self.device)  # [B, #task=64], in 0--59
            if isinstance(mentioned, dict):
                mentioned = {k: torch.stack(v).T.to(self.device) for k, v in mentioned.items()}
            else:
                mentioned = mentioned.to(self.device)  # [B, #task]

            with torch.set_grad_enabled(is_train):
                # pdb.set_trace(header="before fwd")
                logits, loss = model(x, y)
                loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                
                with torch.no_grad(): # just calculating metrics no need for gradients
                    totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    if isinstance(mentioned, dict):
                        for k, mask in mentioned.items():
                            totals_epoch_masks[k] += np.array([torch.sum((age == i) * mask).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    else:
                        totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        totals_epoch_triv += np.array([torch.sum((age == i) * (1 - mentioned)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                
                    y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                    hits = y_hat == y  # [B, #task]
                    hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    if isinstance(mentioned, dict):
                        for k, mask in mentioned.items():
                            hits_masks_epoch[k] += np.array([torch.sum((y_hat == y) * mask * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        for task_idx in range(y.shape[1]):
                            hits_masks_epoch["confusion_matrix"] += sklearn.metrics.confusion_matrix(y[:,task_idx].cpu().numpy(), y_hat[:,task_idx].cpu().numpy(), labels=list(range(num_classes)))
                    else:
                        hits_nontrivial = (y_hat == y) * mentioned
                        hits_trivial = (y_hat == y) * (1 - mentioned)
                        hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        hits_triv_epoch += np.array([torch.sum(hits_trivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        recalls_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        precision_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        total_recalls_epoch += np.array([torch.sum((y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                        total_precision_epoch += np.array([torch.sum((y_hat == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                predictions.append(y_hat.cpu().numpy())

            if is_train:
                # backprop and update the parameters
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                self.optimizer.step()
                mean_loss = float(np.mean(losses))
                mean_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                mean_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                mean_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                    
                lr = self.optimizer.param_groups[0]['lr']

                if isinstance(mentioned, dict):
                    mean_acc_masks = {}
                    report_str = ""
                    for k, mask in mentioned.items():
                        mean_acc_masks[k] = np.sum(hits_masks_epoch[k]).item() / (np.sum(totals_epoch_masks[k]).item() + 1e-20)
                        if "local" in k:
                            report_str += f"{k}: {mean_acc_masks[k] * 100:.2f}%"
                    pbar.set_description(f"Train epoch {epoch + 1}: loss {mean_loss:.5f}; lr {lr:.2e}; acc {mean_acc * 100:.2f}%; {report_str}")
                else:
                    mean_acc_nontriv = np.sum(hits_nontriv_epoch).item() / (np.sum(totals_epoch_nontriv).item() + 1e-20)
                    mean_acc_triv = np.sum(hits_triv_epoch).item() / (np.sum(totals_epoch_triv).item() + 1e-20)
                    pbar.set_description(f"Train epoch {epoch + 1}: loss {mean_loss:.5f}; lr {lr:.2e}; acc {mean_acc * 100:.2f}%; non-triv acc: {mean_acc_nontriv * 100:.2f}; triv acc: {mean_acc_triv * 100:.2f}")
                pbar.refresh()
        if is_train:
            self.train_loss_cont.append(mean_loss)
            self.train_acc_cont.append(mean_acc)
            self.train_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
            if hasattr(data, 'mask_fields'):
                for k, acc in mean_acc_masks.items():
                    self.train_acc_mask_cont[k].append(acc)
            else:
                self.train_acc_nontriv_cont.append(mean_acc_nontriv)
                self.train_acc_triv_cont.append(mean_acc_triv)
        else:  # eval
            test_loss = float(np.mean(losses))
            test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
            if hasattr(data, 'mask_fields'):
                test_acc_masks = {}
                report_str = ""
                for k in hits_masks_epoch.keys():
                    if "confusion_matrix" in k:
                        test_acc_masks[k] = hits_masks_epoch[k]
                    else:
                        test_acc_masks[k] = np.sum(hits_masks_epoch[k]).item() / (np.sum(totals_epoch_masks[k]).item() + 1e-20)
                        report_str += f"{k}: {test_acc_masks[k] * 100:.2f}%; "

                if prt:
                    logger.info(f"confusion matrix:\n{hits_masks_epoch['confusion_matrix']}")
                    logger.info(f"test loss {test_loss:.5f}; acc {test_acc * 100:.2f}%; {report_str}")
            else:
                test_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                test_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / (np.sum(totals_epoch_nontriv).item() + 1e-20)
                test_acc_triv = np.sum(hits_triv_epoch).item() / (np.sum(totals_epoch_triv).item() + 1e-20)
                if prt:
                    # provide recall&precision if binary task
                    msg = f"test loss {test_loss:.5f}; acc {test_acc * 100:.2f}%; non-triv acc {test_acc_nontriv * 100:.2f}%;  test triv acc {test_acc_triv * 100:.2f}%"
                    if self.num_class == 2:
                        msg += f"; test recall {test_recall*100:.2f}%; test precision {test_precision*100:.2f}%"
                    logger.info(msg)
            self.test_loss_cont.append(test_loss)
            self.test_acc_cont.append(test_acc)
            self.test_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
            if hasattr(data, 'mask_fields'):
                for k, acc in test_acc_masks.items():
                    if isinstance(acc, np.ndarray):
                        acc = acc.tolist()
                    self.test_acc_mask_cont[k].append(acc)
            else:
                self.test_acc_nontriv_cont.append(test_acc_nontriv)
                self.test_acc_triv_cont.append(test_acc_triv)

            predictions_matrix = np.concatenate(predictions, axis=0)
            return test_loss, predictions_matrix

    def predict(self, prt=True):
        test_loss, predictions_matrix = self.run_epoch('test', prt)
        return predictions_matrix

    def train(self, prt=True):
        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(self.config.max_epochs):
            self.run_epoch('train', prt, epoch)
            if wandb is not None and wandb.run is not None:
                epoch_log = {"epoch": epoch + 1,
                             "train/loss": self.train_loss_cont[-1],
                             "train/acc": self.train_acc_cont[-1]}
                if self.train_acc_nontriv_cont:
                    epoch_log["train/acc_nontriv"] = self.train_acc_nontriv_cont[-1]
                wandb.log(epoch_log, commit=self.test_dataset is None)
            if self.test_dataset is not None:
                test_loss, predictions_matrix = self.run_epoch('test', prt)
                if wandb is not None and wandb.run is not None:
                    test_log = {"test/loss": test_loss, "test/acc": self.test_acc_cont[-1]}
                    if self.test_acc_nontriv_cont:
                        test_log["test/acc_nontriv"] = self.test_acc_nontriv_cont[-1]
                    wandb.log(test_log)
                self.scheduler.step(test_loss)
                if self.config.debug_train:
                    np.save(f"{self.config.ckpt_path}/predictions_epoch{epoch}.npy", predictions_matrix.astype(int))
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()

        # return predictions after last epoch
        return predictions_matrix

class Mention_Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = f'cuda:{torch.cuda.current_device()}'
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            
        # log something for plotting
        self.train_loss_cont = []
        self.test_loss_cont = []
        self.train_acc_cont = []
        self.test_acc_cont = []
        # would be a list of T-long, each is a lits of MAX_NUM_OPERATIONS-long, for stratified accuracies        
        self.train_strat_acc_cont = []
        self.test_strat_acc_cont = []
        
        
        self.training_history = []
        self.test_history = []
        
    def flush_plot(self, save_path=None):
        # plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi= 80, facecolor='w', edgecolor='k')
        axs = axs.flat
        axs[0].plot(self.train_loss_cont, label="train")
        axs[0].plot(self.test_loss_cont, label="test")
        axs[0].set_title("Loss")
        axs[0].legend()
        axs[1].plot(self.train_acc_cont, label="train")
        axs[1].plot(self.test_acc_cont, label="test")
        axs[1].set_title("Accuracy")
        axs[1].legend()
        if save_path is not None:
            fig.savefig(save_path)
        # plt.show()
        # return a figure object
        

    def save_traces(self, ):
        tbd = {
            "train_loss_cont": self.train_loss_cont, "test_loss_cont" :self.test_loss_cont, 
            "train_acc_cont": self.train_acc_cont, "test_acc_cont": self.test_acc_cont, 
            "train_strat_acc_cont": self.train_strat_acc_cont, "test_strat_acc_cont": self.test_strat_acc_cont, 
        }
        with open(os.path.join(self.config.ckpt_path, "tensorboard.txt"), "w") as f:
            f.write(json.dumps(tbd) + "\n")
        with open(os.path.join(self.config.ckpt_path, "training_history.jsonl"), "w") as f:
            for record in self.training_history:
                f.write(json.dumps(record) + "\n")
        with open(os.path.join(self.config.ckpt_path, "test_history.jsonl"), "w") as f:
            for record in self.test_history:
                f.write(json.dumps(record) + "\n")

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        if not os.path.exists(self.config.ckpt_path):
            os.makedirs(self.config.ckpt_path)
        torch.save(raw_model.state_dict(), os.path.join(self.config.ckpt_path, "checkpoint.ckpt"))

    def load_checkpoint(self):
        torch.load( os.path.join(self.config.ckpt_path, "checkpoint.ckpt"), map_location=self.device)

    def predict(self, prt=True):
        model, config = self.model, self.config
        model.train(False)
        data = self.test_dataset
        loader = DataLoader(data, shuffle=False, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)
        
        pbar = enumerate(loader)
        losses = []
        totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
        hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
        predictions = []
        gt_ls = []
        mentioned_ls = []
        
        results = [] # save predictions for further analysis, if needed
        
        for idx, (x, y, age, mentioned, removed) in pbar:
            gt_ls.append(y.cpu().numpy())
            mentioned_ls.append(mentioned.cpu().numpy())
            
            start_idx, end_idx = idx * config.batch_size, (idx + 1) * config.batch_size
            
            
            x = x.to(self.device)  # [B, f]
            y = y.to(self.device)  # [B, #task=64] 
            age = age.to(self.device)  # [B, #task=64], in 0--59
            mentioned = mentioned.to(self.device) # [B, #task], Bx100
            removed = removed.to(self.device) # [B, #task], Bx100, not used in the model, but can be used for analysis
            # y = mentioned # objective is to classify whether the object is mentioned or not, so just overwrite y with mentioned
            with torch.set_grad_enabled(False):
                logits, loss = model(x, y)
                loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                losses.append(loss.item())
                totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)

                y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                hits = y_hat == y  # [B, #task]
                hits_nontrivial = (y_hat == y) * mentioned
                hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                predictions.append(y_hat.cpu().numpy())
                
        test_loss = float(np.mean(losses))
        test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
        test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()

        if prt: 
            logger.info(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%;  test non-triv acc {test_acc_nontriv*100:.2f}%")
        predictions_matrix = np.concatenate(predictions, axis=0)
        
        # need a confusion matrix for all classes, as well as trivial/non-trivial cases
        # we have prediction matrix (binary or tenary, not convertable), ground truth y(binary or ternary, convertable) and mentioned matrix
        # two cases, if it's a binary probe, we expand the confusion matrix to 2x3, if it's ternary, we just use 3x3
        self.predictions = predictions_matrix
        self.ground_truth = np.concatenate(gt_ls, axis=0)
        self.mentioned = np.concatenate(mentioned_ls, axis=0)
        # self.generate_confusion_matrix(
        #     predictions_matrix,
        #     np.concatenate(gt_ls, axis=0),
        #     np.concatenate(mentioned_ls, axis=0)
        # )
        return predictions_matrix
    
    
    
   
    def train(self, prt=True):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, scheduler = raw_model.configure_optimizers(config)
        
        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=is_train, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            totals_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops of 0 to MAX_NUM_OPERATIONS
            hits_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_nontriv_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_nontriv = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_trivial_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_triv = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_removed_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_removed = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            hits_mention_not_removed_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            totals_epoch_mention_not_removed = np.zeros(MAX_NUM_OPERATIONS, dtype=float)  # np.array of shape [MAX_NUM_OPERATIONS], for numops  of  0 to MAX_NUM_OPERATIONS
            
            
            recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FN)
            precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP / (TP + FP)
            total_recalls_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FN 
            total_precision_epoch = np.zeros(MAX_NUM_OPERATIONS, dtype=float) # TP + FP
            
            predictions = []
            gt_ls = []
            mentioned_ls = []
            pbar = tqdm(enumerate(loader), total=len(loader), disable=not prt) if is_train else enumerate(loader)
            num_mentioned_removed = 0
            num_mentioned_not_removed = 0
            total_mentioned = 0
            results = [] # save predictions for further analysis, if needed
            for it, (x, y, age, mentioned, removed) in pbar: # self.activations[index], self.examples[index], torch.tensor(self.num_ops[index]).to(torch.long), self.mentioned_objects[index]
                x = x.to(self.device)  # [B, f]
                y = y.to(self.device)  # [B, #task=64] 
                age = age.to(self.device)  # [B, #task=64], in 0--59
                mentioned = mentioned.to(self.device) # [B, #task]
                removed = removed.to(self.device)
                mentioned_not_removed = mentioned * (1 - removed)  # [B, #task], only counts the mentioned cases that are not removed
                y = mentioned  # objective is to classify whether the object is mentioned or not, so just overwrite y with mentioned
                start_idx, end_idx = it * config.batch_size, (it + 1) * config.batch_size
                # TODO why when this line is commented out, the acc is much highter? It should be the same as the binary case, but it is not.
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())
                    totals_epoch += np.array([torch.sum(age == i).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_nontriv += np.array([torch.sum((age == i) * mentioned).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_triv += np.array([torch.sum((age == i) * (1 - mentioned)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    totals_epoch_removed += np.array([torch.sum((age == i) * removed).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    y_hat = torch.argmax(logits, dim=-1, keepdim=False)  # [B, #task]
                    expanded_result = [
                        {
                            "index": i + start_idx,
                            "y_hat": y_hat[i].cpu().numpy(),
                            "mentioned": mentioned[i].cpu().numpy(),
                        } for i in range(len(y_hat))
                    ]
                    results.extend(expanded_result)  # save the predictions for further analysis
                    hits = y_hat == y  # [B, #task]
                    # ONLY COUNTS REMOVED CASES
                    hits_removed = (y_hat == y) * removed  # [B, #task], only counts the removed cases
                    hits_removed_epoch += np.array([torch.sum(hits_removed * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    hits_mention_not_removed = (y_hat == y) * mentioned_not_removed  # [B, #task], only counts the mentioned cases that are not removed
                    totals_epoch_mention_not_removed += np.array([torch.sum((age == i) * mentioned_not_removed).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    hits_mention_not_removed_epoch += np.array([torch.sum(hits_mention_not_removed * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    
                    
                    
                    
                    
                    hits_nontrivial = (y_hat == y) * mentioned
                    hits_trivial = hits * (1 - mentioned)  # hits that are trivial (not mentioned)
                    hits_trivial_epoch += np.array([torch.sum(hits_trivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    hits_epoch += np.array([torch.sum(hits * (age == i)).item() for i in range
                                            (MAX_NUM_OPERATIONS)]).astype(float)
                    hits_nontriv_epoch += np.array([torch.sum(hits_nontrivial * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    predictions.append(y_hat.cpu().numpy())
                    gt_ls.append(y.cpu().numpy())
                    mentioned_ls.append(mentioned.cpu().numpy())
                    total_mentioned += np.sum(mentioned.cpu().numpy())
                    num_mentioned_removed += np.sum(removed.cpu().numpy())
                    num_mentioned_not_removed += np.sum(mentioned_not_removed.cpu().numpy())
                    recalls_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    precision_epoch += np.array([torch.sum((y_hat == 1) * (y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_recalls_epoch += np.array([torch.sum((y == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    total_precision_epoch += np.array([torch.sum((y_hat == 1) * (age == i)).item() for i in range(MAX_NUM_OPERATIONS)]).astype(float)
                    # log basic stats of the distributions
               

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    mean_loss = float(np.mean(losses))
                    mean_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                    mean_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()
                    mean_acc_removed = np.sum(hits_removed_epoch).item() / np.sum(totals_epoch_removed).item()
                    mean_acc_mention_not_removed = np.sum(hits_mention_not_removed_epoch).item() / np.sum(totals_epoch_mention_not_removed).item()
                    mean_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                    mean_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                    

                    lr = optimizer.param_groups[0]['lr']
                    pbar.set_description(f"epoch {epoch+1}: train loss {mean_loss:.5f}; lr {lr:.2e}; train acc {mean_acc*100:.2f}%; train non-triv acc: {mean_acc_nontriv*100:.2f}; trivial acc: {np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item()*100:.2f}%; removed acc: {mean_acc_removed*100:.2f}%; mention not removed acc: {mean_acc_mention_not_removed*100:.2f}%, recall {mean_recall*100:.2f}%; precision {mean_precision*100:.2f}%")
                    
                    self.training_history.append({
                        "epoch": epoch+1,
                        "train_loss": mean_loss,
                        "train_acc": mean_acc,
                        "train_nontriv_acc": mean_acc_nontriv,
                        "train_triv_acc": np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item() if np.sum(totals_epoch_triv).item() > 0 else 0.0,
                        "train_recall": mean_recall,
                        "train_precision": mean_precision
                    })
                    
                    
                    
            if is_train:
                self.train_loss_cont.append(mean_loss)
                self.train_acc_cont.append(mean_acc)
                self.train_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
                logger.info(f"Training Total mentioned: {total_mentioned}, num mentioned removed: {num_mentioned_removed}, num mentioned not removed: {num_mentioned_not_removed}")

            if not is_train:
                test_loss = float(np.mean(losses))
                scheduler.step(test_loss)
                test_acc = np.sum(hits_epoch).item() / np.sum(totals_epoch).item()
                test_acc_nontriv = np.sum(hits_nontriv_epoch).item() / np.sum(totals_epoch_nontriv).item()
                test_acc_trivial = np.sum(hits_trivial_epoch).item() / np.sum(totals_epoch_triv).item()
                test_acc_removed = np.sum(hits_removed_epoch).item() / np.sum(totals_epoch_removed).item()
                test_acc_mention_not_removed = np.sum(hits_mention_not_removed_epoch).item() / np.sum(totals_epoch_mention_not_removed).item()
                
                test_recall = np.sum(recalls_epoch).item() / np.sum(total_recalls_epoch).item() if np.sum(total_recalls_epoch).item() > 0 else 0.0
                test_precision = np.sum(precision_epoch).item() / np.sum(total_precision_epoch).item() if np.sum(total_precision_epoch).item() > 0 else 0.0
                logger.info(f"Testing Total mentioned: {total_mentioned}, num mentioned removed: {num_mentioned_removed}, num mentioned not removed: {num_mentioned_not_removed}")
                if prt:                     
                    logger.info(f"test loss {test_loss:.5f}; test acc {test_acc*100:.2f}%;  test non-triv acc {test_acc_nontriv*100:.2f}%;    test trivial acc {test_acc_trivial*100:.2f}%; test removed acc: {test_acc_removed*100:.2f}%; test mention not removed acc: {test_acc_mention_not_removed*100:.2f}%; test recall: {test_recall*100:.2f}%; test precision: {test_precision*100:.2f}%")
                    self.test_history.append({
                    "epoch": epoch+1,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "test_nontriv_acc": test_acc_nontriv,
                    "test_triv_acc": test_acc_trivial,
                    "test_recall": test_recall,
                    "test_precision": test_precision
                })
                self.test_loss_cont.append(test_loss)
                self.test_acc_cont.append(test_acc)
                self.test_strat_acc_cont.append((hits_epoch / totals_epoch).tolist())
                predictions_matrix = np.concatenate(predictions, axis=0)
                self.predictions = predictions_matrix
                self.ground_truth = np.concatenate(gt_ls, axis=0)
                self.mentioned = np.concatenate(mentioned_ls, axis=0)
                
                return test_loss, predictions_matrix

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if wandb is not None and wandb.run is not None:
                wandb.log({"epoch": epoch + 1,
                           "train/loss": self.train_loss_cont[-1],
                           "train/acc": self.train_acc_cont[-1]},
                          commit=self.test_dataset is None)
            if self.test_dataset is not None:
                test_loss, predictions_matrix = run_epoch('test')
                if wandb is not None and wandb.run is not None:
                    test_log = {"test/loss": test_loss, "test/acc": self.test_acc_cont[-1]}
                    if self.test_history:
                        # mention-specific extras recorded per epoch in test_history
                        for k in ("test_nontriv_acc", "test_recall", "test_precision"):
                            if k in self.test_history[-1]:
                                test_log["test/" + k.removeprefix("test_")] = self.test_history[-1][k]
                    wandb.log(test_log)
                if test_loss < best_loss:
                    best_loss = test_loss
                    self.save_checkpoint()

        # return predictions after last epoch
        return predictions_matrix
    
