import json
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_data(dic,k):
    axis = []
    data = []
    for i in dic:
        axis.append(i[k[0]])
        data.append(i[k[1]])
    return axis,data

def plot_accuracy_curves(epochs , train_res, test_res,name='Accuracy'):

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot lines with markers
    plt.plot(epochs, train_res, 'b-o', label=f'Train {name}', linewidth=2, markersize=4)
    plt.plot(epochs, test_res, 'r-o', label=f'Test {name}', linewidth=2, markersize=4)
    
    # Customize the plot
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('Epochs')
    plt.ylabel(f'{name}')
    plt.title(f'Training vs Test {name}')
    plt.legend()
    

    plt.ylim(0.5, 1.0)

    plt.grid(True, linestyle='--', alpha=0.5)
    

    plt.style.use('seaborn-v0_8-whitegrid')
    

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    f = load_json("D:\\projects\\cuda11.8\\3dmodels\\utils\\acc.json")
    axis,train_acc = get_data(f['training_data'],['epoch', 'accuracy'])
    _,test_acc = get_data(f['testing_data'],['epoch', 'accuracy'])
    plot_accuracy_curves(axis,train_acc,test_acc)
    f = load_json("D:\\projects\\cuda11.8\\3dmodels\\utils\\IOU.json")
    axis,train_IOU = get_data(f['training_data'],['epoch', 'IOU'])
    _,test_IOU = get_data(f['testing_data'],['epoch', 'IOU'])
    plot_accuracy_curves(axis,train_IOU,test_IOU,'IOU')