import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    import urllib.request
    from urllib.error import HTTPError

    import os

    dataset_path = "./data"
    checkpoint_path = "./saved_models/tutorial4"
    os.makedirs(checkpoint_path, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial4/"
    pretrained_files = ["FashionMNIST_SGD.config", "FashionMNIST_SGD_results.json", "FashionMNIST_SGD.tar",
                        "FashionMNIST_SGDMom.config", "FashionMNIST_SGDMom_results.json", "FashionMNIST_SGDMom.tar",
                        "FashionMNIST_Adam.config", "FashionMNIST_Adam_results.json", "FashionMNIST_Adam.tar"]

    for file_name in pretrained_files:
        file_path = os.path.join(checkpoint_path, file_name)
        if not os.path.isfile(file_path):
            file_url = base_url + file_name
            print(f"Donwloading {file_url}...")
            try:
                urllib.request.urlretrieve(file_url, file_path)
            except HTTPError as e:
                print("Something went wrong.")

    from torchvision.datasets import FashionMNIST
    from torchvision import transforms

    # Transformations applied on each image -> first make them a tensor and normalize
    trf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = FashionMNIST(root=dataset_path, train=True, transform=trf, download=True)
    train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
    # Loading the test set
    test_set = FashionMNIST(root=dataset_path, train=False, transform=trf, download=True)

    import torch.utils.data as data

    train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False)
    val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)
    test_loader = data.DataLoader(test_set, batch_size=1024, shuffle=False, drop_last=False)

    print("Mean", (train_dataset.data.float() / 255.0).mean().item())
    print("Std", (train_dataset.data.float() / 255.0).std().item())

    imgs, _ = next(iter(train_loader))
    print(f"Mean: {imgs.mean().item():5.3f}")
    print(f"Standard deviation: {imgs.std().item():5.3f}")
    print(f"Maximum: {imgs.max().item():5.3f}")
    print(f"Minimum: {imgs.min().item():5.3f}")

    import torch.nn as nn


    class BaseNetwork(nn.Module):
        def __init__(self, act_fn, input_size=784, num_classes=10, hidden_size=[512, 256, 256, 128]):
            super().__init__()

            # Create the network based on the specified hidden sizes
            layers = []
            layer_sizes = [input_size] + hidden_size
            for layer_index in range(1, len(layer_sizes)):
                layers += [nn.Linear(layer_sizes[layer_index - 1], layer_sizes[layer_index]), act_fn]
            layers += [nn.Linear(layer_sizes[-1], num_classes)]
            self.layers = nn.ModuleList(
                layers)  # A module list registers a list of modules as submodules (e.g. for parameters)

            self.config = {"act_fn": act_fn.__class__.__name__, "input_size": input_size, "num_classes": num_classes,
                           "hidden_sizes": hidden_size}

        def forward(self, x):
            x = x.view(x.size(0), -1)
            for l in self.layers:
                x = l(x)
            return x


    class Identity(nn.Module):
        def forward(self, x):
            return x


    act_fn_by_name = {
        "tanht": nn.Tanh,
        "relu": nn.ReLU,
        "identity": Identity
    }

    ## Initialization

    model = BaseNetwork(act_fn=Identity()).to(device)
    def const_init(model, c = 0.0):
        for name, param in model.named_parameters():
            param.data.fill_(c)

    const_init(model, c=0.005)
    