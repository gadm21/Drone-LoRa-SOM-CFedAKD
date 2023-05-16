import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# a function that calculates keras model size in KB
def get_model_size(model):
    size = model.count_params() * 4 / 1e3
    return size

# a function that calculates numpy array size in KB
def get_array_size(array):
    size = array.nbytes / 1e3
    return size

def size_of(obj) : 
    # if obj is a numpy array
    if isinstance(obj, np.ndarray) :
        return get_array_size(obj)
    else: 
        return get_model_size(obj)
    



def aggregate(soft_labels, compress) : 
    soft_labels = np.array(soft_labels)
    
    if compress : 
        n_sl = soft_labels - soft_labels.min(axis = 1, keepdims = True)
        n_sl = n_sl / n_sl.max(axis = 1, keepdims = True)
        n_sl = n_sl * 255
        n_sl = n_sl.astype(np.uint8)
        global_sl = np.mean(n_sl, axis = 0).astype(np.uint8)
    else : 
        global_sl = np.mean(soft_labels, axis = 0)
    
    
    return global_sl 


def show_dataset_samples(classes, samples_per_class, 
                         images, labels, data_type="MNIST"):
    num_classes = len(classes)
    fig, axes = plt.subplots(samples_per_class, num_classes, 
                             figsize=(num_classes, samples_per_class)
                            )
    
    for col_index, cls in enumerate(classes):
        idxs = np.flatnonzero(labels == cls)
        idxs = np.random.choice(idxs, samples_per_class, 
                                replace=False)
        for row_index, idx in enumerate(idxs):    
            if data_type == "MNIST":
                axes[row_index][col_index].imshow(images[idx],
                                                  cmap = 'binary', 
                                                  interpolation="nearest")
                axes[row_index][col_index].axis("off")
            elif data_type == "CIFAR":
                axes[row_index][col_index].imshow(images[idx].astype('uint8'))
                axes[row_index][col_index].axis("off")
                
            else:
                print("Unknown Data type. Unable to plot.")
                return None
            if row_index==0:
                axes[row_index][col_index].set_title("Class {0}".format(cls))
                
                
    plt.show()
    return None



# def plot_history(model):
    
#     """
#     input : model is trained keras model.
#     """
    
#     fig, axes = plt.subplots(2,1, figsize = (12, 6), sharex = True)
#     axes[0].plot(model.history.history["loss"], "b.-", label = "Training Loss")
#     axes[0].plot(model.history.history["val_loss"], "k^-", label = "Val Loss")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()
    
    
#     axes[1].plot(model.history.history["acc"], "b.-", label = "Training Acc")
#     axes[1].plot(model.history.history["val_acc"], "k^-", label = "Val Acc")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("Accuracy")
#     axes[1].legend()
    
#     plt.subplots_adjust(hspace=0)
#     plt.show()
    
# def show_performance(model, Xtrain, ytrain, Xtest, ytest):
#     y_pred = None
#     print("CNN+fC Training Accuracy :")
#     y_pred = model.predict(Xtrain, verbose = 0).argmax(axis = 1)
#     print((y_pred == ytrain).mean())
#     print("CNN+fc Test Accuracy :")
#     y_pred = model.predict(Xtest, verbose = 0).argmax(axis = 1)
#     print((y_pred == ytest).mean())
#     print("Confusion_matrix : ")
#     print(confusion_matrix(y_true = ytest, y_pred = y_pred))
    
#     del y_pred