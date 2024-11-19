import torch
import numpy as np
import time
import copy
import torch.nn as nn
from system.src.optimizers.fedoptimizer import PerturbedGradientDescent
from system.src.clients.clientBase import Client

from sklearn.metrics import accuracy_score,precision_score
from sklearn.preprocessing import label_binarize




class clientSR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.mu = args.mu

        self.lamda = args.lamda

        self.global_model = copy.deepcopy(self.model)
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.last_local_model = copy.deepcopy(self.model)
        for param in self.last_local_model.parameters():
            param.data.zero_()

        self.loss = nn.CrossEntropyLoss()
        # self.optimizer = PerturbedGradientDescent(
        #     self.model.parameters(), lr=self.learning_rate, mu=self.mu)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )


    def train(self, round):
        # innovation1
        # if round > 1:
        #     local_model_F_bata = evaluate_model(self.args, self.load_test_data(), self.local_model)
        #     global_model_F_bata = evaluate_model(self.args, self.load_test_data(), self.model)
        #     if local_model_F_bata > global_model_F_bata:
        #         self.model = self.local_model
        #################################

        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                # self.optimizer.step(self.global_params, self.device)
                self.optimizer.step()

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # innovation
        self.last_local_model = copy.deepcopy(self.model)


    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)

                gm = torch.cat([p.data.view(-1) for p in self.global_model.parameters()], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                lm = torch.cat([p.data.view(-1) for p in self.last_local_model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm-pm, p=2)
                loss -= 0.5 * self.lamda * torch.norm(gm-lm, p=2)
                # ro = 0.3
                # loss += 0.5 * self.mu * torch.norm(gm-pm, p=2) * (1 - ro) + torch.abs(gm-pm).sum()

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

def evaluate_model(args, testloader, model):
    bata = 1

    model.eval()
    y_prob = []
    y_true = []

    with torch.no_grad():
        for x, y in testloader:
            if type(x) == type([]):
                x[0] = x[0].to(args.device)
            else:
                x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)

            prob = torch.sigmoid(output).cpu().numpy()

            y_prob.append(prob)
            nc = args.num_classes
            if args.num_classes == 2:
                nc += 1
            lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
            if args.num_classes == 2:
                lb = lb[:, :2]
            y_true.append(lb)

    y_prob = np.concatenate(y_prob, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    max_indices = np.argmax(y_prob, axis=1)
    y_pred = np.zeros_like(y_prob)
    for i in range(len(max_indices)):
        y_pred[i][max_indices[i]] = 1

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')

    F_bata = ((1 + bata ** 2) * acc * precision) / ((bata ** 2 * acc) + precision)

    return F_bata
