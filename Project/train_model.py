import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt


def train_model(dataloader_train,
                dataloader_val,
                optimizer,
                net,
                num_epochs,
                device,
                create_plot=False,
                class_weights=None,
                print_every_50_batches = False,
                save_best_model_as = None):

    if class_weights is not None:
      class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss_history = []
    val_loss_history = []
    epoch_loss_history = []
    every_50_batches_loss_history = []

    best_loss = float("inf")

    for epoch in range(num_epochs):
        val_loss = 0

        net.train()

        i = 0
        for features, labels in tqdm.tqdm(dataloader_train):
            i+=1
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # clear the gradients
            outputs = net(features)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            loss.backward()  # compute the gradients
            optimizer.step()  # tweak weights
            loss_value = loss.item()
            loss_history.append(loss_value)
            if i % 50 == 0 and print_every_50_batches and len(dataloader_val) != 0:
              every_50_batches_loss_history.append(loss_history[-1])
              net.eval()
              with torch.no_grad():
                  for features, labels in dataloader_val:
                      features = features.to(device)
                      labels = labels.to(device)

                      outputs = net(features)
                      loss = criterion(outputs, labels)
                      val_loss += loss.item()
              val_loss /= len(dataloader_val)

              val_loss_history.append(val_loss)
              epoch_loss_history.append(loss_history[-1])
              net.train()

              if val_loss < best_loss and save_best_model_as is not None:
                torch.save(net.state_dict(), save_best_model_as)


        if len(dataloader_val) != 0 and print_every_50_batches == False:
            net.eval()
            with torch.no_grad():
                for features, labels in dataloader_val:
                    features = features.to(device)
                    labels = labels.to(device)

                    outputs = net(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= len(dataloader_val)

            val_loss_history.append(val_loss)
            epoch_loss_history.append(loss_history[-1])

    if create_plot:
        train_loss = np.array(loss_history)
        val_loss = np.array(val_loss_history)
        epochs = np.arange(num_epochs)
        epoch_loss = np.array(epoch_loss_history)
        every_50_batches_loss = np.array(every_50_batches_loss_history)

        if print_every_50_batches:
          num_ticks = val_loss.shape[0]
          ticks = np.arange(num_ticks)
          plt.plot(ticks, val_loss, label="Validation Loss")
          plt.plot(ticks, every_50_batches_loss, label= "Train Loss")
        elif len(dataloader_val) != 0:
            plt.plot(epochs, val_loss, label="Validation Loss")
            plt.plot(epochs, epoch_loss, label= "Train Loss")
        else:
            plt.plot(np.arange(train_loss.shape[0]),
                    train_loss,
                    label="Train Loss")  
        plt.legend()     
        plt.xlabel("Epoch")
        plt.ylabel("Loss (Cross-Entropy)")
        plt.ylim(0, np.max(train_loss) * 1.05)
        plt.show()

        if len(dataloader_val) != 0 and print_every_50_batches==False:

          plt.plot(epochs, epoch_loss, label="Train Loss")
          plt.plot(epochs, val_loss, label="Validation Loss")

          plt.xlabel("Epoch")
          plt.ylabel("Loss (Cross-Entropy)")
          plt.ylim(0, np.max(val_loss) * 1.05)
          plt.title("Loss (scaled by VAL max)")
          plt.legend()
          plt.show()



    return loss_history

