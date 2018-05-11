import re
import matplotlib.pyplot as plt

if __name__=='__main__':
    file = open('./logs/log.txt','r')
    accuracy = []
    epochs = []
    loss = []
    for line in file:
        test_accuracy = re.search('"test_accuracy": ([0]\.[0-9]+)*', line)
        if test_accuracy:
            accuracy.append(test_accuracy.group(1))
        
        epoch = re.search('"epoch": ([0-9]+)*', line)
        if epoch:
            epochs.append(epoch.group(1))
        
        train_loss = re.search('"train_loss": ([0-9]\.[0-9]+)*', line)
        if train_loss:
            loss.append(train_loss.group(1))
    file.close()
    plt.figure('test_accuracy vs epochs')  
    plt.xlabel('epoch')
    plt.ylabel('test_accuracy')
    plt.plot(epochs,accuracy,'b*')
    plt.plot(epochs,accuracy,'r')
    plt.grid(True)

    plt.figure('train_loss vs epochs')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.plot(epochs,loss,'b*')
    plt.plot(epochs,loss,'y')
    plt.grid(True)

    plt.show()
    
