import pandas as pd  # Імпорт бібліотеки для роботи з даними у вигляді таблиць
import matplotlib.pyplot as plt  # Імпорт бібліотеки для візуалізації даних
from collections import Counter  # Імпорт функції для лічильника
from sklearn import metrics  # Імпорт метрик для оцінки якості кластеризації
from sklearn.cluster import KMeans  # Імпорт алгоритму KMeans для кластеризації
from sklearn.preprocessing import MinMaxScaler  # Імпорт методу MinMaxScaler для нормалізації даних
from tabulate import tabulate  # Імпорт функції для створення табличного виведення даних

# Завантаження даних з CSV-файлу
def load_glass_data(file_path):
    return pd.read_csv(file_path)

# Попередня обробка та кластеризація даних
def preprocess_and_cluster_glass_data(data):
    # TASK2: Визначення ознак та меток
    features = data.iloc[:, :-1].values  # Виділення ознак (колонок з першої до передостанньої)
    target = data.iloc[:, 9].values  # Виділення міток (колонка "Type")

    # Масштабування ознак методом Min-Max
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    # K-Means кластеризація
    clusterer = KMeans(n_clusters=6)
    clusterer.fit(scaled_features)
    labels = clusterer.labels_
    metrics.silhouette_score(scaled_features, labels, metric="euclidean")
    predictions = clusterer.predict(scaled_features)
    data["cluster"] = predictions  # Додавання колонки "cluster" з результатами кластеризації
    print(data)

    # TASK3: Визначення центроїд та побудова графіків розсіювання
    centroids = clusterer.cluster_centers_  # Визначення центроїд кожного кластера
    print("Centroids: ")
    print(centroids, "\n")

    # Функція для побудови графіків розсіювання
    def create_scatter_plot(index1, index2):
        fig, ax = plt.subplots()
        scatter1 = ax.scatter(scaled_features[:, index1], scaled_features[:, index2],
                              c=predictions, s=18, cmap="Reds")
        handles, labels = scatter1.legend_elements()
        legend1 = ax.legend(handles, labels, loc="lower right")
        ax.add_artist(legend1)
        scatter2 = ax.scatter(centroids[:, index1], centroids[:, index2], marker="x",
                              c="yellow", s=200, linewidth=2, label="centroids")
        plt.legend(loc="upper left")
        plt.xlabel(f"{data.columns[index1]}")
        plt.ylabel(f"{data.columns[index2]}")
        plt.show()

    create_scatter_plot(0, 1)
    create_scatter_plot(2, 3)
    create_scatter_plot(4, 5)
    create_scatter_plot(6, 7)

    # TASK4: Аналіз кластерів
    count_clusters = Counter(labels)  # Підрахунок кількості об'єктів у кожному кластері
    print("Objects in clusters: ")
    print(count_clusters, "\n")
    cluster_content = data.groupby(["cluster", "Type"]).size().unstack(fill_value=0)
    cluster_content["Total"] = cluster_content.sum(axis=1)
    cluster_content.loc["Total"] = cluster_content.sum()
    print(tabulate(cluster_content, headers="keys", tablefmt="psql"))

    # TASK5: Аналіз та візуалізація оптимальної кількості кластерів
    df = pd.DataFrame(columns=["Number of clusters", "WCSS", "Silhouette", "DB"])
    for i in range(2, 11):
        clusterer_i = KMeans(n_clusters=i).fit(scaled_features)
        predictions_i = clusterer_i.predict(scaled_features)
        WCSS = clusterer_i.inertia_  # Внутрішньокластерна сума квадратів відстаней
        Silhouette = metrics.silhouette_score(scaled_features, predictions_i)  # Силует
        DB = metrics.davies_bouldin_score(scaled_features, predictions_i)  # Індекс Davies-Bouldin
        new_row_df = pd.DataFrame([[i, WCSS, Silhouette, DB]], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))

    def build_method_graphic(method_name):
        plt.plot(df["Number of clusters"], df[method_name], marker="o", linestyle="None", label=method_name, color="red")
        plt.xlabel("Number of clusters")
        plt.ylabel(method_name + " method")
        plt.title(method_name + " method")
        plt.legend()
        plt.show()

    build_method_graphic("WCSS")
    build_method_graphic("Silhouette")
    build_method_graphic("DB")

    # TASK6: Кластеризація з масштабуванням
    clusterer = KMeans(n_clusters=6)
    clusterer.fit(features)
    labels = clusterer.labels_
    metrics.silhouette_score(features, labels, metric="euclidean")
    predictions = clusterer.predict(features)
    data["cluster"] = predictions

    count_clusters = Counter(labels)
    print("Objects in clusters: ")
    print(count_clusters, "\n")

    cluster_content = data.groupby(["cluster", "Type"]).size().unstack(fill_value=0)
    cluster_content["Total"] = cluster_content.sum(axis=1)
    cluster_content.loc["Total"] = cluster_content.sum()
    print(tabulate(cluster_content, headers="keys", tablefmt="psql"))

    df = pd.DataFrame(columns=["Number of clusters", "WCSS", "Silhouette", "DB"])
    for i in range(2, 11):
        clusterer_i = KMeans(n_clusters=i).fit(scaled_features)
        predictions_i = clusterer_i.predict(scaled_features)
        WCSS = clusterer_i.inertia_
        Silhouette = metrics.silhouette_score(scaled_features, predictions_i)
        DB = metrics.davies_bouldin_score(scaled_features, predictions_i)
        new_row_df = pd.DataFrame([[i, WCSS, Silhouette, DB]], columns=df.columns)
        df = pd.concat([df, new_row_df], ignore_index=True)

    print(tabulate(df, headers="keys", tablefmt="psql", floatfmt=".3f"))

    build_method_graphic("WCSS")
    build_method_graphic("Silhouette")
    build_method_graphic("DB")

# Виклик функції
file_path = "Glass.csv"
glass_data = load_glass_data(file_path)
print(glass_data)
preprocess_and_cluster_glass_data(glass_data)
