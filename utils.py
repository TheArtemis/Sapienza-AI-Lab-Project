import torch
import torchvision
import sklearn.metrics as mt
import numpy as np
import pandas as pd

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer'])

def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2*(preds*y).sum()) / ((preds+y).sum()+1e-8)

    print(f'Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}')
    print(f'Dice score: {dice_score/len(loader)}')
    model.train()

def save_predictions_as_imgs(loader, model, folder='imgs/', device='cuda'):
    model.eval()    
    for idx, (x, y) in enumerate(loader):

        if idx > 30: # Save only the first 20 predictions
            break

        x = x.to(device=device)        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )

    model.train()


# A function to calculate all the metrics we need with sklearn 

def evaluate_model(loader, model, loss_fn, device='cuda'):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    acc = mt.accuracy_score(y_true, y_pred)
    precision = mt.precision_score(y_true, y_pred)
    recall = mt.recall_score(y_true, y_pred)
    f1 = mt.f1_score(y_true, y_pred)
    loss = loss_fn(torch.tensor(y_pred), torch.tensor(y_true).float()).item()   
    
    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')
    print(f'Loss: {loss:.2f}')    

    model.train()
    return acc, precision, recall, f1, loss

def get_val_loss(loader, model, loss_fn, device='cuda'):
    model.eval()
    val_loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            y_true.append(y.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    val_loss = loss_fn(torch.tensor(y_pred), torch.tensor(y_true).float()).item()   
    model.train()
    return val_loss

def update_losses(df, loader, model, loss_fn, model_name, epoch, train_loss):
    val_loss = get_val_loss(loader, model, loss_fn)
    new_row = {'model': model_name, 'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
    df = df.append(new_row, ignore_index=True)
    return df

def update_metrics(df, loader, model, loss_fn, model_name, epoch, train_loss, device='cuda'):
    acc, prec, rec, f1, val_loss = evaluate_model(loader, model, loss_fn, device)

    new_row = {'model': model_name, 'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    df = df.append(new_row, ignore_index=True)
    return df     


# function to test evaluate_model
def test():
    y_true = np.array([[1, 0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0]]).flatten()
    y_pred = np.array([[1, 1, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1, 0]]).flatten()

    acc = mt.accuracy_score(y_true, y_pred)
    precision = mt.precision_score(y_true, y_pred)
    recall = mt.recall_score(y_true, y_pred)
    f1 = mt.f1_score(y_true, y_pred)

    print(f'Accuracy: {acc:.2f}')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 score: {f1:.2f}')

if __name__ == '__main__':
    test()
