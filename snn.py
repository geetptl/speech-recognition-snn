import argparse
import dataset as D
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import network as N
# import matplotlib.pyplot as plt
# import numpy as np


def run(path):
    print("Loading data...")
    dataset = D.load(path)
    print(f"{len(dataset)} samples")

    def snn_training(network, spike_ts, device, batch_size=128, test_batch_size=256, epoch=100):
        train_dataset, test_dataset = random_split(dataset, [0.7, 0.3])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.7)

        train_loss_list, test_accuracy_list = [], []
        test_num = len(test_dataset)
        network.to(device)

        for ee in range(epoch):
            running_loss = 0.0
            running_batch_num = 0

            for data in train_dataloader:
                image, label = data
                image = image.to(device)
                label = label.to(device)
                event_image = N.generate_spike_signatures(image, device, spike_ts)

                optimizer.zero_grad()
                output = network(event_image)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_batch_num += 1
            train_loss_list.append(running_loss / running_batch_num)
            print(f"epoch {ee}")
            print(f"\ttraining loss {train_loss_list[-1]:.2f}")

            test_correct_num = 0
            with torch.no_grad():
                for data in test_dataloader:
                    image, label = data
                    image = image.to(device)
                    label = label.to(device)
                    event_image = N.generate_spike_signatures(image, device, spike_ts)

                    outputs = network(event_image)
                    _, predicted = torch.max(outputs, 1)
                    test_correct_num += (predicted == label).sum().item()
            test_accuracy_list.append(test_correct_num / test_num)
            print(f"\taccuracy {test_accuracy_list[-1]:.2f}")
        network.to("cpu")
        return train_loss_list, test_accuracy_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    param_dict = {"hid_layer": [0.5, 0.6, 0.4, 0.95, 0.8], "out_layer": [0.5, 0.6, 0.4, 0.95, 0.8]}

    snn = N.SNN_Network(9152, 10, 256, param_dict, device)

    train_loss_list, test_accuracy_list = snn_training(
        network=snn, spike_ts=5, device=device, batch_size=32, test_batch_size=64, epoch=15
    )

    # fig, ax1 = plt.subplots()

    # color1 = 'tab:orange'
    # ax1.set_xlabel('epoch')
    # ax1.set_ylabel('Training Loss', color=color1)
    # ax1.set_xticks(range(0, len(train_loss_list)))
    # ax1.plot(train_loss_list, color=color1)
    # ax1.tick_params(axis='y')
    # ax1.grid(visible=True, which='major', axis='y', color=color1, alpha=0.4)
    # ax2 = ax1.twinx()

    # color2 = 'tab:blue'
    # ax2.set_ylabel('Accuracy', color=color2)
    # ax2.plot(test_accuracy_list, color=color2)
    # ax2.grid(visible=True, which='major', axis='y', color=color2, alpha=0.4)
    # ax2.tick_params(axis='y')

    # fig.tight_layout()
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        default="./.data/recordings/",
        type=str,
        help="Path to the downloaded data files",
    )
    args = parser.parse_args()

    run(args.path)
