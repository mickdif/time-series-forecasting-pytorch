#da https://www.alphavantage.co/stock-prediction-deep-neural-networks-lstm/
#In this project, we will train an LSTM model to predict stock price movements. 


from os import write
import numpy as np #NumPy is the fundamental package for scientific computing in Python. 
#It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), 
#and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, 
#selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.


import torch # PyTorch is a Python package that provides two high-level features:
#Tensor computation (like NumPy) with strong GPU acceleration
#Deep neural networks built on a tape-based autograd system
import torch.nn as nn # These are the basic building blocks for graph
import torch.nn.functional as F # Convolution functions
import torch.optim as optim # is a package implementing various optimization algorithms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. 
# Matplotlib makes easy things easy and hard things possible. 
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from alpha_vantage.timeseries import TimeSeries

print("All libraries loaded")

# def file scrittura
f = open('workfile.txt', 'w', encoding="utf-8")

disegna=0

config = {
    "alpha_vantage": {
        "key": "demo", # you can use the demo API key for this project, but please make sure to get your own API key at https://www.alphavantage.co/support/#api-key
        "symbol": "IBM",
        "outputsize": "full", # MODIFICATO era full
        "key_adjusted_close": "5. adjusted close", # note that we are using the adjusted close field of Alpha Vantage's daily 
        #adjusted API to remove any artificial price turbulences due to stock splits and dividend payout events. 
        #It is generally considered an industry best practice to use split/dividend-adjusted prices instead of raw prices to 
        #model stock price movements. 
        # per IBM ci sono i dati dal 99
    },
    "data": {
        "window_size": 20,
        "train_split_size": 0.50,
    },
    "plots": {
        "xticks_interval": 90, # show a date every 90 days
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 1, # since we are only using 1 feature, close price
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu", # "cuda" or "cpu"
        "batch_size": 5, # +++ MOD erano 64
        "num_epoch": 5, # +++ MOD erano 100
        "learning_rate": 0.1,# +++ MOD era 0.01
        "scheduler_step_size": 40,
    }
}

def download_data(config):
    ts = TimeSeries(key=config["alpha_vantage"]["key"])
    data, meta_data = ts.get_daily_adjusted(config["alpha_vantage"]["symbol"], outputsize=config["alpha_vantage"]["outputsize"])
    # vd https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=IBM&apikey=demo per capire la struttura dei dati
    data_date = [date for date in data.keys()] # la funzione keys() per qualsiasi ogg di python restituisce la lista degli argomenti dell'ogg
    data_date.reverse()

    data_close_price = [float(data[date][config["alpha_vantage"]["key_adjusted_close"]]) for date in data.keys()]
    # se ne deriva che data è stato restituito come matrice la cui prima riga è composta dalle date, e le successive dai valori di apertura chiusura etc, una riga ciascuno
    # data_close_price è dunque la lista di tutti i valori di chiusura aggiustati, perché dire "per ciascuna data" è come dire tutti
    data_close_price.reverse()
    data_close_price = np.array(data_close_price) # lo trasformo in array di numpy

    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1] # è una stringa
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range = download_data(config)

# DEBUG
# print("Data close price")
# for i in range(2):
#     print(data_close_price)


    # plot
if(disegna==1):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close price for " + config["alpha_vantage"]["symbol"] + ", " + display_date_range)
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    # salva su disco senza mostrare
    # plt.show()
    fig.savefig('Fig_1.png', transparent=False, dpi=80, bbox_inches="tight")


print("Ora normalizzo")
class Normalizer():
    # una gaussiana ha 2 parametri: mu (indica la media, il centro, x del picco) e la variazione std
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x): 
        self.mu = np.mean(x, axis=(0), keepdims=True) # mean() calcola la media dei dati sull'asse specificato
        self.sd = np.std(x, axis=(0), keepdims=True) # deviazione std
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

# normalize


np.savetxt("data_close_price.txt", np.round(data_close_price,2))
scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

np.savetxt("normalized_data_close_price.txt", np.round(normalized_data_close_price,2))

# ora i valori sono normalizzati, prima andavano da 50 a 150, ora ballano tra, circa, -1,9 e 2,1

# DEBUG
print("normalized data close price example")
for i in range(2):
    print(normalized_data_close_price)
    
def prepare_data_x(x, window_size): # x=normalized_data_close_price, windows_size=20
    # perform windowing
    n_row = x.shape[0] - window_size + 1
    # shape[0] in questo caso equivalente a lenght
    # n_row corrisponde al numero di dati, nel caso di ibm abbiamo ciascun giorno lavorativa dal 99 a oggi, circa 6000 giorni
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    # shape lo rende una matrice di n_row righe e windows_size colonne
    # strides non so cosa faccia
    # in ogni riga della matrice ci sono i prezzi di chiusura adj degli ultimi 20 giorni
    # ad esempio
    # sia x=[1,2,3,4]
    # output=[[1,2]
    #         [2,3]
    #         [3,4]]
    # a ogni riga si scorre di uno
    # occhio che una scelta sbagliata dei valori crea caos perché si cerca di inserire dati che non esistono
    return output[:-1], output[-1]
    # [:-1] toglie l'ultimo valore
    # [-1] restituisce solo l'ultimo valore


def prepare_data_y(x, window_size):
    # perform simple moving average
    # output = np.convolve(x, np.ones(window_size), 'valid') / window_size

    # use the next day as label
    output = x[window_size:]
    # è una lista uguale ad x ma eliminati i primi window_size elementi
    return output

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
# data_x è una matrice 2d che contiene i prezzi degli ultimi 20 giorni degli ultimi 23 anni (da quando iniziano i dati, per IBM dal 99), tolto oggi
# data_x_unseen era l'ultima riga, quindi i prezzi degli ultimi 20 giorni relativamente a oggi
data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])
# è uguale a normalized_data_close_price priva dei primi window_size (=20) elementi

# DEBUG
print("data_y")
print(data_x_unseen)


# split dataset

split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]

data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

# prepare data for plotting

to_plot_data_y_train = np.zeros(num_data_points)
to_plot_data_y_val = np.zeros(num_data_points)

to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

# plots
if(disegna==1):
    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, to_plot_data_y_train, label="Prices (train)", color=config["plots"]["color_train"])
    plt.plot(data_date, to_plot_data_y_val, label="Prices (validation)", color=config["plots"]["color_val"])
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.title("Daily close prices for " + config["alpha_vantage"]["symbol"] + " - showing training and validation data")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
    fig.savefig('Fig_2.png', transparent=False, dpi=80, bbox_inches="tight")


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        x = np.expand_dims(x, 2)  # in our case, we have only 1 feature, so we need to convert `x` into [batch, sequence, features] for LSTM
        self.x = x.astype(np.float32) # cambia il tipo
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])


dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True) # DataLoader è di PyThorch
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        batchsize = x.shape[0]

        # layer 1
        x = self.linear_1(x)
        x = self.relu(x)

        # LSTM layer
        lstm_out, (h_n, c_n) = self.lstm(x)

        # reshape output from hidden cell into [batch, features] for `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1)

        # layer 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]


def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += (loss.detach().item() / batchsize)

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

# le due righe qua di seguito sono le stesse che si trovano prima di delle def di classe e di fz qua sopra, non ne capisco l'utilità ma non sono dannose
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)


for epoch in range(config["training"]["num_epoch"]):
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()

    print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))
    f.write('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f} | lr:{:.6f}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train))


# here we re-initialize dataloader so the data doesn't shuffled, so we can plot the values by date
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

# predict on the training data, to see how well the model managed to learn and memorize

predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_train = np.concatenate((predicted_train, out))

# predict on the validation data, to see how the model does

predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val = np.concatenate((predicted_val, out))



if(disegna==1):
    # prepare data for plotting

    to_plot_data_y_train_pred = np.zeros(num_data_points)
    to_plot_data_y_val_pred = np.zeros(num_data_points)

    to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
    to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

    to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

    # plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(data_date, data_close_price, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(data_date, to_plot_data_y_train_pred, label="Predicted prices (train)", color=config["plots"]["color_pred_train"])
    plt.plot(data_date, to_plot_data_y_val_pred, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Compare predicted prices to actual prices")
    xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)] # make x ticks nice
    x = np.arange(0,len(xticks))
    plt.xticks(x, xticks, rotation='vertical')
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
    fig.savefig('predicted prices vs. actual prices.png', transparent=False, dpi=80, bbox_inches="tight")

    # prepare data for plotting the zoomed in view of the predicted prices vs. actual prices

    to_plot_data_y_val_subset = scaler.inverse_transform(data_y_val)
    to_plot_predicted_val = scaler.inverse_transform(predicted_val)
    to_plot_data_date = data_date[split_index+config["data"]["window_size"]:]

    # plots

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(to_plot_data_date, to_plot_data_y_val_subset, label="Actual prices", color=config["plots"]["color_actual"])
    plt.plot(to_plot_data_date, to_plot_predicted_val, label="Predicted prices (validation)", color=config["plots"]["color_pred_val"])
    plt.title("Zoom in to examine predicted price on validation data portion")
    xticks = [to_plot_data_date[i] if ((i%int(config["plots"]["xticks_interval"]/5)==0 and (len(to_plot_data_date)-i) > config["plots"]["xticks_interval"]/6) or i==len(to_plot_data_date)-1) else None for i in range(len(to_plot_data_date))] # make x ticks nice
    xs = np.arange(0,len(xticks))
    plt.xticks(xs, xticks, rotation='vertical')
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
    fig.savefig('predicted prices vs. actual zoomed.png', transparent=False, dpi=80, bbox_inches="tight")


# +++++++++++++++++++++++++++++++++++++++++++++++++
# predict the closing price of the next trading day

model.eval()

x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2) # this is the data type and shape required, [batch, sequence, feature]
prediction = model(x)
prediction = prediction.cpu().detach().numpy()


if(disegna==1):
    # prepare plots

    plot_range = 10
    to_plot_data_y_val = np.zeros(plot_range)
    to_plot_data_y_val_pred = np.zeros(plot_range)
    to_plot_data_y_test_pred = np.zeros(plot_range)

    to_plot_data_y_val[:plot_range-1] = scaler.inverse_transform(data_y_val)[-plot_range+1:]
    to_plot_data_y_val_pred[:plot_range-1] = scaler.inverse_transform(predicted_val)[-plot_range+1:]

    to_plot_data_y_test_pred[plot_range-1] = scaler.inverse_transform(prediction)
    # fa una conversione da array a scalare, occhio che è deprecato in numpy!

    to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)
    to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)
    to_plot_data_y_test_pred = np.where(to_plot_data_y_test_pred == 0, None, to_plot_data_y_test_pred)

    # plot

    plot_date_test = data_date[-plot_range+1:]
    plot_date_test.append("tomorrow")

    fig = figure(figsize=(25, 5), dpi=80)
    fig.patch.set_facecolor((1.0, 1.0, 1.0))
    plt.plot(plot_date_test, to_plot_data_y_val, label="Actual prices", marker=".", markersize=10, color=config["plots"]["color_actual"])
    plt.plot(plot_date_test, to_plot_data_y_val_pred, label="Past predicted prices", marker=".", markersize=10, color=config["plots"]["color_pred_val"])
    plt.plot(plot_date_test, to_plot_data_y_test_pred, label="Predicted price for next day", marker=".", markersize=20, color=config["plots"]["color_pred_test"])
    plt.title("Predicting the close price of the next trading day")
    plt.grid(visible=None, which='major', axis='y', linestyle='--')
    plt.legend()
    plt.show()
    fig.savefig('tomorrow.png', transparent=False, dpi=80, bbox_inches="tight")

print("Predicted close price of the next trading day:", round(scaler.inverse_transform(prediction), 2)) # forse

f.write("Predicted close price of the next trading day:", round(to_plot_data_y_test_pred[plot_range-1], 2))
f.close()
print("FINE CIAO")