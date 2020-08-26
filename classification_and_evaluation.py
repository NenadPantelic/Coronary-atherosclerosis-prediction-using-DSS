from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from seaborn import heatmap


def apply_algorithm(training_data, test_data, classification_algorithm):
    # assert "train_x" in data and "train_y" in data and "test_x" in data and "test_y" in data, "Input data must contain test samples."
    classifier = classification_algorithm.fit(training_data["x"], training_data["y"])
    predicted_output = classifier.predict(test_data["x"])
    return predicted_output


def calculate_accuracy(y, y_predicted):
    return round(accuracy_score(y, y_predicted), 2)


def calculate_precision(y, y_predicted):
    return round(precision_score(y, y_predicted), 2)


def calculate_roc_curve(y, y_predicted):
    # false positives, true positives
    fpr, tpr, _ = roc_curve(y, y_predicted)
    # ROC
    roc_auc = round(auc(fpr, tpr), 2)
    return fpr, tpr, roc_auc


def calculate_f1(y, y_predicted):
    return round(f1_score(y, y_predicted, average="micro"), 2)


def calculate_confusion_matrix(y, y_predicted):
    # confusion matrix
    return confusion_matrix(y, y_predicted)


def calculate_and_plot_metrics(algorithm_name, y, y_predicted):
    ax1, ax2 = new_plot_fig() # TODO: every time new fig window is opend - recode it
    precision = calculate_precision(y, y_predicted)
    accuracy = calculate_accuracy(y, y_predicted)
    f1 = calculate_f1(y, y_predicted)
    false_pos, true_pos, roc_auc = calculate_roc_curve(y, y_predicted)
    conf_matrix = calculate_confusion_matrix(y, y_predicted)
    print(f"*******{algorithm_name}*******")
    print_metrics_data(auc=roc_auc, precision=precision, accuracy=accuracy, f1_score=f1)
    plot_roc_curve(ax1, false_pos, true_pos)
    plot_confusion_matrix(ax2, conf_matrix)
    plt.suptitle(algorithm_name, fontsize=18)
    plt.show()


def print_metrics_data(**kwargs):
    print("---------")
    for key, value in kwargs.items():
        print(f"{key} = {value}")
    print("---------")


# plotting
def new_plot_fig():
    fig = plt.figure(figsize=(8, 2.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    return ax1, ax2


def plot_roc_curve(ax, false_pos, true_pos):
    ax.plot(false_pos, true_pos, color="darkorchid")
    ax.plot([0, 1], [0, 1], color="dodgerblue")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve", loc="center")


def plot_confusion_matrix(ax, conf_matrix):
    heatmap(conf_matrix, annot=True, cmap="BuPu", ax=ax)
    ax.set_title("Confusion matrix", loc='center')
