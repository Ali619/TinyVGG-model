import torch
from torch import nn 
from TinyVGG import tinyVGG
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from train_test_loop_function import train_loop, test_loop
from accuracy_fn import accuracy_fn 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Downlaoding FashionMNIST dataset
train_data = FashionMNIST(root='./FashionMNIST',train=True, download=True, transform=ToTensor(), target_transform=None)
test_data = FashionMNIST(root='./FashionMNIST', train=False, download=True, transform=ToTensor())

class_names = train_data.classes
print(f"\nclass names are: {class_names}\n length of classes: {len(class_names)}\n")

# turning data into batches
from torch.utils.data import DataLoader
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

# Creating the model (Creating an object from TinyVGG class)
model_1 = tinyVGG(input_shape=1, hidden_state=10, ouput_shape=len(class_names))
print(f"The model stracture is: \n{model_1}")

# Setting the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(lr=0.1, params=model_1.parameters())

# Now let's train the model for 3 epochs
epochs = 3
print("Training is start\n")
for epoch in range(epochs):
    print(f"Epoch: {epoch+1}\n--------")

    train_loop(model=model_1,
                data=train_dataloader,
                optimizer=optimizer,
                loss=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)
    
    test_loop(model=model_1,
                data=test_dataloader,
                loss=loss_fn,
                accuracy_fn=accuracy_fn,
                device=device)

print("\nTraining is Done")