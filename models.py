import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from tqdm import tqdm_notebook as tqdm

# Helper function to convert numpy array to torch if necessary
def to_torch(X):
    if type(X) == np.ndarray: 
        X = torch.from_numpy(X)
    return X

# Helper class for training and testing PyTorch modules
class FitModule(nn.Module):
    def fit(self, 
            X, Y, 
            loss_fn, 
            eval_fn = None,
            val_data = None, 
            batch_size = 32, 
            epochs = 10, 
            optimizer = torch.optim.Adam, 
            filepath = None,
            cuda = 1,
            verbose=1):
        
        X, Y = to_torch(X), to_torch(Y)
        
        if val_data:
            Xtest, Ytest = val_data
            Xtest, Ytest = to_torch(Xtest), to_torch(Ytest)
        
        opt = optimizer([p for p in self.parameters() if p.requires_grad])

        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        if val_data:
            val_loss = self.evaluate(Xtest, Ytest, loss_fn, batch_size=batch_size, cuda=cuda)
            if eval_fn:
                val_eval = self.evaluate(Xtest, Ytest, eval_fn, batch_size=batch_size, cuda=cuda)
            filtered_loss = val_loss
        else:
            filtered_loss = self.evaluate(X, Y, loss_fn, batch_size=batch_size, cuda=cuda)
        
        with tqdm(total=epochs*len(loader)) as pbar:
            for epoch in range(epochs):
                self.train()
                for batch_idx, (x, y) in enumerate(loader):
                    if cuda:
                        x, y = x.cuda(), y.cuda()

                    x, y = Variable(x), Variable(y)

                    opt.zero_grad()
                    z = self.forward(x)
                    loss = loss_fn(z,y)
                    loss.backward()
                    opt.step()
                    
                    filtered_loss = 0.95 * filtered_loss + 0.05 * loss.data.item() # changed from data[0]
                    
                    pbar.set_description("Epoch: %i/%i" %(epoch+1, epochs))
                    if val_data:
                        if eval_fn:
                            pbar.set_postfix(loss = filtered_loss, val_loss=val_loss, val_eval=val_eval)
                        else:
                            pbar.set_postfix(loss = filtered_loss, val_loss=val_loss)
                    else:
                        pbar.set_postfix(loss = filtered_loss)

                    pbar.update(1)

                if val_data:
                    val_loss = self.evaluate(Xtest, Ytest, loss_fn, batch_size=batch_size, cuda=cuda)
                    if eval_fn:
                        val_eval = self.evaluate(Xtest, Ytest, eval_fn, batch_size=batch_size, cuda=cuda)
                        pbar.set_postfix(loss = filtered_loss, val_loss=val_loss, val_eval=val_eval)
                    else:
                        pbar.set_postfix(loss = filtered_loss, val_loss=val_loss)

                if filepath:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        torch.save(self, filepath)

    def predict(self, X, batch_size=32, cuda=1, verbose=0, memmap = None):
        X = to_torch(X)
        
        dataset = torch.utils.data.TensorDataset(X, torch.Tensor(X.size()[0]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if verbose:
            pbar = tqdm(total=len(loader))

        self.eval()
        r, n = 0, X.size()[0]
        for batch_idx, batch_data in enumerate(loader):
            if cuda:
                batch_data[0] = batch_data[0].cuda()

            X_batch = Variable(batch_data[0])
            y_batch_pred = self(X_batch).data

            if r == 0:
                y_pred = torch.zeros((n,) + y_batch_pred.size()[1:])

            y_pred[r : min(n, r + batch_size)] = y_batch_pred
            r += batch_size

            if verbose:
                pbar.set_description("Batch: %i/%i" %(batch_idx+1, len(loader)))
                pbar.update(1)

        return y_pred
    
    def predict_memmap(self, X, batch_size=32, cuda=1, verbose=0, memmap = None):
        X = to_torch(X)

        dataset = torch.utils.data.TensorDataset(X, torch.Tensor(X.size()[0]))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if verbose:
            pbar = tqdm(total=len(loader))

        self.eval()
        r, n = 0, X.size()[0]
        for batch_idx, batch_data in enumerate(loader):
            if cuda:
                batch_data[0] = batch_data[0].cuda()

            X_batch = Variable(batch_data[0])
            y_batch_pred = self(X_batch).data

            if r == 0:
                shape = (n,) + y_batch_pred.size()[1:]
                if memmap == None:
                    y_pred = np.zeros(shape)
                else:
                    y_pred = np.memmap(memmap, dtype='float32', mode='w+', shape=shape)

            y_pred[r : min(n, r + batch_size)] = y_batch_pred.cpu().numpy()
            r += batch_size

            if verbose:
                pbar.set_description("Batch: %i/%i" %(batch_idx+1, len(loader)))
                pbar.update(1)

        return y_pred
    
    def evaluate(self, X, Y, eval_fn, batch_size=32, cuda=1, verbose=0):
        X, Y = to_torch(X), to_torch(Y)
        
        dataset = torch.utils.data.TensorDataset(X, Y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        if verbose:
            pbar = tqdm(total=len(loader))

        evaluations = torch.zeros(len(loader))
        batch_sizes = torch.zeros(len(loader))
        
        self.eval()
        for batch_idx, (x, y) in enumerate(loader):
            if cuda:
                x, y = x.cuda(), y.cuda()
            
            x, y = Variable(x), Variable(y)

            z = self.forward(x)

            evaluations[batch_idx] = eval_fn(z, y).data
            batch_sizes[batch_idx] = x.size(0)

            if verbose:
                pbar.set_description("Batch: %i/%i" %(batch_idx+1, len(loader)))
                pbar.update(1)
        
        result = torch.sum(evaluations * batch_sizes) / torch.sum(batch_sizes)
        return result.item()

# Sequential version of FitModule
class FitSequential(nn.Sequential):
    fit = FitModule.fit
    predict = FitModule.predict
    evaluate = FitModule.evaluate

# Local response normalization layer
# Adapted from code by Jiecao Yu: https://github.com/pytorch/pytorch/issues/653#issuecomment-326851808 
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

# Flatten operation layer
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Normalization of images for ImageNet CNNs
class Normalize(nn.Module):
    def forward(self, x):
        mean=[0.4914, 0.4822, 0.4465]
        std =[0.2023, 0.1994, 0.2010]
        z = x.clone()
        for i in range(3):
            z[:,i] = (z[:,i]-mean[i]) / std[i]

        return z

class CNN(nn.Module):
    def __init__(self, num_classes=1000):
        super(CNN, self).__init__()
        
        NL = nn.Sequential(nn.ReLU(), LRN(15))
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, padding=4), nn.BatchNorm2d(32), NL,
            nn.MaxPool2d(kernel_size=2, stride=2), # 112x112

            nn.Conv2d(32, 48, kernel_size=9, padding=4, stride=1), nn.BatchNorm2d(48), NL,
            nn.MaxPool2d(kernel_size=2, stride=2), # 56x56

            nn.Conv2d(48, 96, kernel_size=7, padding=3, stride=1), nn.BatchNorm2d(96), NL,
            nn.MaxPool2d(kernel_size=2, stride=2), # 28x28

            nn.Conv2d(96, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), NL,
            nn.MaxPool2d(kernel_size=2, stride=2), # 14x14

            nn.Conv2d(192, 384, kernel_size=5, padding=2), nn.BatchNorm2d(384), NL,
            nn.MaxPool2d(kernel_size=2, stride=2), # 7x7

            nn.Conv2d(384, 768, kernel_size=5, padding=2), nn.BatchNorm2d(768), NL,
            nn.Conv2d(768, 768, kernel_size=5, padding=2), nn.BatchNorm2d(768), NL,
            nn.Conv2d(768, 768, kernel_size=5, padding=2), nn.BatchNorm2d(768), NL,
            nn.AvgPool2d(7), # 1x1
        )
        
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Multiplicative Gaussian noise
class MGaussianNoise(nn.Module):
    def __init__(self, sigma):
        super(MGaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.autograd.Variable(x.data.new(x.size()).normal_(1.0, self.sigma))
        return x * noise

# Loads trained CNN and optionally adds multiplicative Gaussian noise layers after each Conv2D layer 
def load_trained_model(filepath, keep_classifier=False, add_noise_layers=False, noise_sigma=0.15):
    net = CNN(num_classes=1000)
    net.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
    
    model = FitSequential(Normalize(), *list(net.features)[:-1])

    if add_noise_layers:
        _model = []
        for l in model:
            _model.append(l)
            if type(l).__name__ == 'Conv2d':
                _model.append(MGaussianNoise(noise_sigma))
        
        model = FitSequential(*_model)

    if keep_classifier:
        model = FitSequential(*list(model)[1:], net.features[-1],  Flatten(), net.classifier, nn.Softmax(-1))

    return model