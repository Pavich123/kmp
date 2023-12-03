import matplotlib.pyplot as plt #підключення необхідних бібліотек

import pandas as pd
from pandas.plotting import scatter_matrix

from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

import numpy as np
from sklearn.svm import SVC




def first(my_set):
    print("Statistical info:")
    print(my_set.describe(), "\n")#вивід статистичної інформації

    print("====================================================================")
    print("Class distribution by ", my_set.groupby('Type').size(), "\n")#розподілення по класам
    
    df = pd.DataFrame(my_set)#створюємо dataframe
    non = df.isna()#витягуємо пропущені значення та дублікати

    print("====================================================================")
    print("DataFrame with missed values:")
    print(df)#dataframe із пропущеними значеннями

    print("====================================================================")
    print("\nWhat is missed?:")
    print(non)#вивід пропущених значень та дублікатів

    print("====================================================================")
    print("\nNot null values in the columns:")
    print(df.info())#вивід непорожніх значень у кожній колонці
    


def second(my_set):
    COLORS = {
        "1": "blue",
        "2": "green",
        "3": "red",
        "5": "purple",
        "6": "grey",
        "7": "yellow"
    }#список кольорів
    
    diff_colors = my_set["Type"].map(lambda x: COLORS.get(x))#надаємо кольори
    scatter_matrix(my_set, c=diff_colors)#будуємо матрицю
    
    plt.show()#вивід на екран


def DATUN(my_set):
    my_types = my_set.iloc[:, :-1].values #отримаємо type
    features = my_set.iloc[:, 9].values #отримаємо ознаки
    train_size, test_size, train_class, test_class = train_test_split(my_types, features, test_size=0.2)#отримуємо тренувальні і тестові дані
    return train_size, test_size, train_class, test_class #повертаємо значення





def third_fourth_fifth(my_set): 
    train_size, test_size, train_class, test_class = DATUN(my_set) #отримуємо значення
    classifier = SVC()#створення класифікатору
    classifier.fit(train_size, train_class)#надаємо тренувальний набор значень
    correct = classifier.score(test_size, test_class)#правильні передбачення
    print("Correct predictions proportion = ", correct, "\n")

    print("============================================================")
    predict = classifier.predict(test_size)#передбачення на основі тестових даних
    print(classification_report(test_class, predict))#виводимо оцінку значень класифікатора

    print("==========================================================")
    cm = confusion_matrix(test_class, predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)#створення матриці
    disp.plot()#малювання матрциі
    plt.show()#вивід на екран


def SIZE(train_size, test_size):#функція масштабування
    scaler = StandardScaler()
    scaler.fit(train_size)#задаємо size величину для масштабування
    train_size = scaler.transform(train_size)#трансформація
    test_size = scaler.transform(test_size)
    return train_size, test_size#повертання

def sixth(my_set):
    train_size, test_size, train_class, test_class = DATUN(my_set)#отримуємо значення
    train_size, test_size = SIZE(train_size, test_size)#функція масштабування
    
    classifier = SVC()#створення класифікатору
    classifier.fit(train_size, train_class)#надаємо тренувальний набор значень
    correct = classifier.score(test_size, test_class)#правильні передбачення
    print("Correct predictions proportion = ", correct, "\n")

    print("==========================================================")
    predict = classifier.predict(test_size)#передбачення на основі тестових даних
    print(classification_report(test_class, predict))#виводимо оцінку значень класифікатора

    print("==========================================================")
    cm = confusion_matrix(test_class, predict)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)#створення матриці
    disp.plot()#малювання матрциі
    plt.show()#вивід на екран


def seventh(my_set):
    train_size, test_size, train_class, test_class = DATUN(my_set)#отримуємо значення
    train_size, test_size = SIZE(train_size, test_size)#функція масштабування

    # Створіть класифікатор з заданими параметрами
    classifier = SVC(C=1.0, gamma='scale', kernel='rbf')
    
    classifier.fit(train_size, train_class)#надаємо тренувальний набор значень
    correct = classifier.score(test_size, test_class)#правильні передбачення
    print("Correct predictions proportion = ", correct, "\n")



    
def euclidean_distance(point1, point2):
    #Розрахунок евклідової відстані між двома точками
    return np.sqrt(np.sum((point1 - point2)**2))

def find_nearest_neighbor(test_object, training_set):
    #Знаходження найближчого сусіда для тестового об'єкта в навчальному наборі
    distances = [euclidean_distance(test_object, train_object) for train_object in training_set]
     # Знаходження індексу найменшої відстані (найближчого сусіда)
    nearest_neighbor_index = np.argmin(distances)
    return nearest_neighbor_index



def last(my_set):
    training_set, test_objects, train_data, test_data = DATUN(my_set)#отримуємо тренувальні і тестові дані
    # Знаходимо найближчого сусіда для кожного тестового об'єкта
    for i, test_object in enumerate(test_objects):
        nearest_neighbor_index = find_nearest_neighbor(test_object, training_set)
    
        print(f"Тестовий об'єкт {i+1}: {test_object}")
        print(f"Найближчий сусід: {training_set[nearest_neighbor_index]}")
    
        # Порівняння класів тестового об'єкта та його найближчого сусіда
        if nearest_neighbor_index == i:
            print("Класи співпадають.")
        else:
            print("Класи не співпадають.")
    
        print("-" * 30)


def main():
    my_set = pd.read_csv("Glass.csv")#створюємо цифровий dataset
    my_set["Type"] = [str(i) for i in my_set["Type"]]#розподіляємо по категоріях
    #first(my_set)
    #second(my_set)
    #third_fourth_fifth(my_set)
    sixth(my_set)
    #seventh(my_set)
    #last(my_set)


main()
