
""" Run this. If it doesn't work, contact me."""


import torch
import torch.nn as nn  # neuron network module
import torch.nn.functional as F  # neuron network functions
import torch.optim as optim

import numpy as np

import torchvision
import torchvision.transforms as transforms

from timeit import timeit
import matplotlib.pyplot as plt
import cv2

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class ImageNet64(nn.Module):
    def __init__(self, classes, training=False):
        super(ImageNet64, self).__init__()
        self.training = training

        self.classes = classes

        w1, w2, w3 = 64, 32, 24
        p1, p2, p3 = 5, 3, 2

        n1, n2 = 240, 168

        self.pool = nn.MaxPool2d(2, 2)   # Max pooling over a (2, 2) window
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, w1, 2*p1+1, padding_mode='zeros', padding=p1)
        self.conv2 = nn.Conv2d(w1, w2, 2*p2+1, padding_mode='zeros', padding=p2)
        self.conv3 = nn.Conv2d(w2, w3, 2*p3+1, padding_mode='zeros', padding=p3)
        self.conv1_bn = nn.BatchNorm2d(w1)
        self.conv2_bn = nn.BatchNorm2d(w2)
        self.conv3_bn = nn.BatchNorm2d(w3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(w3 * 4 * 4, n1)  # 5*5 from image dimension
        self.fc2 = nn.Linear(n1, n2)
        self.fc1_bn = nn.BatchNorm1d(n1)
        self.fc2_bn = nn.BatchNorm1d(n2)

        self.fc3 = nn.Linear(n2, 10)

    def dropout(self, x):
        return F.dropout(x, training=self.training, p=0.1)

    def forward(self, x):

        # print(f"Input: {x.shape}")
        x = self.dropout(self.pool(F.relu(self.conv1_bn(self.conv1(x))))) # shape: 3 * 32 * 32 -> conv 5 * 5 -> 6 * 28 * 28 -> maxpool 2 * 2 -> 6 * 14 * 14
        # print(f"conv1: {x.shape}")
        x = self.dropout(self.pool(F.relu(self.conv2_bn(self.conv2(x)))))  # shape: 6 * 14 * 14 -> conv 5 * 5 -> 16 * 10 * 10 -> maxpool 2 * 2 -> 16 * 5 * 5
        # print(f"conv2: {x.shape}")
        x = self.dropout(self.pool(F.relu(self.conv3_bn(self.conv3(x)))))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # print(f"flatten: {x.shape}")
        x = F.relu(self.fc1_bn(self.fc1(x)))
        # print(f"linear1: {x.shape}")
        x = F.relu(self.fc2_bn(self.fc2(x)))
        # print(f"linear2: {x.shape}")
        # x = self.fc3(x)
        x = F.softmax(self.fc3(x), dim=1)
        # print(f"linear3 (output): {x.shape}")
        return x

    def validate(self, testloader, categoty_test=False):
        self.eval()
        correct_preds, total_tests = 0, 0

        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}

        for i, data in enumerate(testloader):
            self.training = False

            X_batch, y_batch = data

            y_pred = self.predict(X_batch)

            if categoty_test:
                for label, prediction in zip(y_batch, y_pred):
                    if label == prediction:
                        correct_pred[self.classes[label]] += 1
                    total_pred[self.classes[label]] += 1

            correct_preds += torch.sum(y_pred == y_batch).item()
            total_tests += y_batch.shape[0]

        if categoty_test:
            for classname, correct_count in correct_pred.items():
                accuracy = 100 * float(correct_count) / total_pred[classname]
                print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

        print(correct_preds, total_tests)
        return correct_preds / total_tests
    def predict(self, X_test, show_likelyhood = False):
        self.eval()
        self.training = False
        with torch.no_grad():
            y_pred = self(X_test)
        if show_likelyhood: return y_pred
        return torch.argmax(y_pred, dim=1)

    def set(self, criterion=None, optimizer=None):
        self.criterion = criterion
        self.optimizer = optimizer
    def my_train(self, epochs=None, trainloader=None, testloader=None):
        self.train()
        print(f'Start training.')
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            for i, data in enumerate(trainloader):
                self.training = True

                X_batch, y_batch = data

                self.optimizer.zero_grad()

                y_pred = self(X_batch)  # prediction = forward pass

                loss = self.criterion(y_pred, y_batch)

                # print(loss.grad_fn)  # MSELoss
                # print(loss.grad_fn.next_functions[0][0])  # Linear
                # print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

                # print(f'conv1.bias.grad before backward: {net.conv1.bias.grad}')
                loss.backward()  # gradients = backward pass # dl/dw
                # print(f'conv1.bias.grad after backward: {net.conv1.bias.grad}')

                self.optimizer.step()  # update weights

                running_loss += loss.item()
                if i and not i % max(1, len(trainloader) // 5):
                    average_loss = running_loss / (i)
                    print(f'ep: {epoch}, batch: {i}, loss: {average_loss:.3f}')

            if not epoch % max(1, epochs // 10):
                average_loss = running_loss / len(trainloader)
                print(f'epoch: {epoch}, loss: {average_loss:.3f}')
        accuracy = self.validate(testloader)
        print(f'Finished Training, accuracy = {accuracy:.3f}')


    def prepare_first_layer_display(self):
        x0_mesh = x1_mesh = np.arange(0, 5, 1)
        x0_mesh, x1_mesh = np.meshgrid(x0_mesh, x1_mesh)

        mesh_shape, mesh_width = x0_mesh.shape, x0_mesh.shape[0]

        x0s, x1s = np.ravel(x0_mesh), np.ravel(x1_mesh)
        x0s, x1s = torch.from_numpy(x0s), torch.from_numpy(x1s)
        x = torch.cat((x0s, x1s), 0)
        x = torch.reshape(x, (2, -1))
        x = torch.transpose(x, 0, 1)

        weights, bias = list(self.conv1.parameters())
        weights = weights.detach().numpy()
        bias = bias.detach().numpy()


        colormap = weights[:, [1, 2, 0], :, :]  # rgb -> brg

        # normalization
        # colormap = np.clip(colormap, 0, 100)
        norm = np.amax(np.abs(colormap))
        print('norm', norm)
        colormap = colormap / norm / 2 + 0.5

        # reshape
        colormap = np.transpose(colormap, [0, 2, 3, 1])
        colormap = np.flip(colormap, axis=1)
        # print(colormap[0])

        return colormap


def train_model():
    net = ImageNet64(classes)

    net.set(criterion=nn.CrossEntropyLoss(), optimizer=optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=5e-5))
    # optimizer = optim.Adam(net.parameters(), lr=0.005, weight_decay=5e-7)

    print(f'training time: {timeit(lambda: net.my_train(EPOCHS, trainloader, testloader), number=1)}')

    PATH = './path_name.pth'
    torch.save(net.state_dict(), PATH)


def test_any_image():
    net = ImageNet64(classes)
    PATH = './path_name.pth'
    net.load_state_dict(torch.load(PATH))

    img_list, n_img = []
    for i in range(n_img):
        img_path = f'.img{i}.jpg'
        image_data = cv2.imread(img_path)  # Read an image
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(image_data, (32, 32))  # Resize to the same size as

        image_data = transform(image_data)
        img_list.append(image_data)

        image_data = torch.unsqueeze(image_data, 0)

        prediction = classes[net.predict(image_data)]  # Get label name from label index
        print(prediction)

    img_list = torch.stack(img_list, 0)
    imshow(torchvision.utils.make_grid(img_list, nrow=8))


def test_in_testset():
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    net = ImageNet64(classes)
    PATH = './path_name.pth'
    net.load_state_dict(torch.load(PATH))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    count = 0
    list_of_correct = []
    lable_string = ''
    predicted_string = ''
    for j in range(batch_size_test):
        if labels[j] != predicted[j]:
            list_of_correct.append(images[j])
            lable_string += f"{classes[labels[j]]}, "
            predicted_string += f"{classes[predicted[j]]}, "
            count += 1
        if count == 32: break
    correct = torch.stack(list_of_correct, 0)

    imshow(torchvision.utils.make_grid(correct, nrow=16))
    print(lable_string)
    print(predicted_string)

    print(images.shape)
    print(correct.shape)


def validate():
    net = ImageNet64(classes)
    PATH = './path_name.pth'

    net.load_state_dict(torch.load(PATH))
    print(net.validate(testloader, categoty_test=True))



"""downloading and loading image"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    EPOCHS, batch_size, batch_size_test = 10, 128, 128

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=2)

    print(f'Images loaded.')

    # call what you need

    # train_model()
    #
    # validate()
    #
    # test_in_testset()
    #
    # test_any_image()











