import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import random
import matplotlib.pyplot as plt
import sklearn.utils

# --- Гиперпараметры ---
POPULATION_SIZE = 5
GENERATIONS = 20
MUTATION_RATE = 0.1

# --- Загрузка данных ---
df = pd.read_csv("KBTU-Data-Camp/kbtu-data-science-challenge-2025-entry-task-new/train.csv")

# --- Подготовка данных ---
df = df.drop(columns=["age", "gender"])  # Игнорируем ненужные колонки

# One-hot encoding для категориальных переменных
df = pd.get_dummies(df, columns=["parental_education", "school_type"], drop_first=True)

def prepare_data():
    # Разделение признаков и целевой переменной
    X = df.drop(columns=["final_math_score"])
    y = df["final_math_score"]
    
    # Перемешивание данных перед разбиением
    X, y = sklearn.utils.shuffle(X, y, random_state=random.randint(0, 10000))
    
    # Разделение на тренировочный и тестовый набор
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Нормализация (Z-score scaling)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def build_model(input_dim):
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),  # Явно задаем входной слой
        layers.Dense(8, activation="relu"),
        layers.Dense(4, activation="relu"),
        layers.Dense(1)  # Выходной слой без активации
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

# --- Функция создания популяции ---
def create_population(size, input_dim):
    return [build_model(input_dim) for _ in range(size)]

# --- Функция оценки ---
def evaluate_population(population, X_train, X_test, y_train, y_test):
    scores = []
    for model in population:
        model.fit(X_train, y_train, epochs=10, verbose=1, batch_size=40)  # Короткое обучение
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        scores.append((mse, model))
    return scores

# --- Селекция (выбор лучших) ---
def select_best(scores, top_k):
    scores.sort(key=lambda x: x[0])  # Сортируем по MSE (чем меньше, тем лучше)
    return [model for _, model in scores[:top_k]]

# --- Кроссовер (усреднение весов) ---
def crossover(parent1, parent2):
    weights1 = parent1.get_weights()
    weights2 = parent2.get_weights()
    new_weights = [(w1 + w2) / 2 for w1, w2 in zip(weights1, weights2)]
    child = build_model(len(new_weights[0]))  # Создаем новую модель с нужным входным размером
    child.set_weights(new_weights)
    return child

# --- Мутация ---
def mutate(model):
    weights = model.get_weights()
    new_weights = [w + np.random.rand() - 0.5 if np.random.rand() < MUTATION_RATE else w for w in weights]
    model.set_weights(new_weights)
    return model

# --- Основной процесс эволюционного обучения ---
mse_history = []

X_train, X_test, y_train, y_test = prepare_data()
input_dim = X_train.shape[1]
population = create_population(POPULATION_SIZE, input_dim)

for generation in range(GENERATIONS):
    print(f"Generation {generation+1}")

    X_train, X_test, y_train, y_test = prepare_data()
    # Оценка популяции
    scores = evaluate_population(population, X_train, X_test, y_train, y_test)
    best_mse = min(scores, key=lambda x: x[0])[0]
    mse_history.append(best_mse)
    print(f"Best mse: {best_mse:.4f}, rmse: {np.sqrt(best_mse):.4f}")
    
    # Выбор лучших
    best_models = select_best(scores, 2)
    
    # Создание нового поколения
    p1, p2 = random.sample(best_models, 2)
    new_population = []
    while len(new_population) < POPULATION_SIZE:
        child = crossover(p1, p2)
        child = mutate(child)
        new_population.append(child)
    
    population = new_population

# --- Построение графика MSE ---
plt.plot(range(1, GENERATIONS + 1), mse_history, marker='o', linestyle='-')
plt.xlabel("Generation")
plt.ylabel("Best MSE")
plt.title("MSE Evolution Over Generations")
plt.grid()
plt.show()