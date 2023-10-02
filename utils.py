import torch
import torchvision

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
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(
        f'Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}'
    )

def save_predictions_as_imgs(loader, model, folder='imgs/', device='cuda'):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(
            y.unsqueeze(1), f'{folder}/mask_{idx}.png'
        )
        #print(y.shape)
