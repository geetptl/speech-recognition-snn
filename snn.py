import argparse
import dataset as D
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import network as N


def run(path):
    print("Loading data...")
    dataset = D.load(path)
    print(f"{len(dataset)} samples")

    def stbp_snn_training(network, spike_ts, device, batch_size=128, test_batch_size=256, epoch=100):
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
                event_image = N.img_2_event_img(image, device, spike_ts)

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
                    event_image = N.img_2_event_img(image, device, spike_ts)

                    outputs = network(event_image)
                    _, predicted = torch.max(outputs, 1)
                    test_correct_num += (predicted == label).sum().item()
            test_accuracy_list.append(test_correct_num / test_num)
            print(f"accuracy {test_accuracy_list[-1]:.2f}")
        network.to("cpu")
        return train_loss_list, test_accuracy_list

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    param_dict = {"hid_layer": [0.5, 0.6, 0.4, 0.95], "out_layer": [0.5, 0.6, 0.4, 0.95]}

    snn = N.WrapSNN(9152, 10, 256, param_dict, device)

    train_loss_list, test_accuracy_list = stbp_snn_training(
        network=snn, spike_ts=5, device=device, batch_size=32, test_batch_size=64, epoch=20
    )


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
