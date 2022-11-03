import torch

def train_loop(model: torch.nn.Module, 
                data: torch.utils.data.DataLoader,
                optimizer: torch.optim.Optimizer,
                loss: torch.nn.Module,
                accuracy_fn,
                device: torch.device):

    model.to(device)
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data):

        X_train, y_train = X.to(device), y.to(device)

        model.train()
        y_preds = model(X_train)

        _loss = loss(y_preds, y_train)
        train_loss += _loss

        train_acc += accuracy_fn(y_train, y_preds.argmax(dim=-1))

        optimizer.zero_grad()

        _loss.backward()

        optimizer.step()

    train_loss /= len(data)
    train_acc /= len(data)

    print(f"Train Loss: {train_loss:.5f}, Train Accuracy %{train_acc:.2f}")

def test_loop(model: torch.nn.Module,
                data: torch.utils.data.DataLoader,
                loss: torch.nn.Module,
                accuracy_fn,
                device: torch.device):
    
    model.to(device)
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data:

            X_test, y_test = X.to(device), y.to(device)

            y_preds = model(X_test)

            test_loss = loss(y_preds, y_test)

            test_acc += accuracy_fn(y_test, y_preds.argmax(dim=-1))
        
        test_loss /= len(data)
        test_acc /= len(data)

    print(f"Test Loss: {test_loss:.5f}, Test Accuracy: %{test_acc:.2f}\n")