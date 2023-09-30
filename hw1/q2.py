from sklearn import datasets
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

#loads iris and plots a boxplot for each feature by species
def load_iris():

    iris = datasets.load_iris()
    col_list = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    iris_dataframe = pandas.DataFrame(data = iris.data, columns = col_list)

    #target_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}


    iris_dataframe["target"] = iris.target
    #iris_dataframe["target"] = iris_dataframe["target"].map(target_names)
    # target_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    # iris_dataframe["target"] = pandas.Categorical.from_codes(iris.target, categories=target_names.values(), ordered=False)
    # iris_dataframe["target"] = iris_dataframe["target"].astype("series")

    boxplot = iris_dataframe.boxplot(column=col_list, by="target")
    plt.suptitle("Boxplots of Iris Features by Species")

    plt.show()
    return iris_dataframe




def sepal_scatter_plot():
    iris = datasets.load_iris()
    col_list = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    iris_dataframe = pandas.DataFrame(data = iris.data, columns = col_list)
    iris_dataframe["target"] = iris.target

    x_length = iris_dataframe["sepal length (cm)"]
    y_width = iris_dataframe["sepal width (cm)"]
    color_names = iris_dataframe["target"]
    color_names = {0: "Red", 1: "Blue", 2: "Green"}

    iris_dataframe["target"] = iris_dataframe["target"].map(color_names)  # Map target labels to plant names

    plt.suptitle("Scatterplots of Iris Sepal Features by Species")
    plt.xlabel("Length")
    plt.ylabel("Width")

    plt.scatter(x_length, y_width, c=iris_dataframe["target"])
    plt.show()

def petal_scatter_plot():
    iris = datasets.load_iris()
    col_list = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    iris_dataframe = pandas.DataFrame(data = iris.data, columns = col_list)
    iris_dataframe["target"] = iris.target

    x_length = iris_dataframe["petal length (cm)"]
    y_width = iris_dataframe["petal width (cm)"]
    color_names = iris_dataframe["target"]
    color_names = {0: "Red", 1: "Blue", 2: "Green"}

    iris_dataframe["target"] = iris_dataframe["target"].map(color_names)  # Map target labels to plant names

    plt.suptitle("Scatterplots of Iris Petal Features by Species")
    plt.xlabel("Length")
    plt.ylabel("Width")

    plt.scatter(x_length, y_width, c=iris_dataframe["target"])
    plt.show()


load_iris()
sepal_scatter_plot()
petal_scatter_plot()
