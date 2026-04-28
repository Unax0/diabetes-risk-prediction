import streamlit as st
import joblib
import numpy as np

# Загрузка модели (обученной без удалённых фичей)
model = joblib.load("xgb_model_reduced.pkl")

st.title("Опросник: Прогноз риска диабета")

st.subheader("Ответьте на вопросы")

# Ввод возраста
age_input = st.number_input("Введите ваш возраст", min_value=0, max_value=150, value=18)

# Преобразование возраста в категорию и отображение диапазона
if 18 <= age_input <= 24:
    Age = 1
    age_range = "18-24 лет"
elif 25 <= age_input <= 29:
    Age = 2
    age_range = "25-29 лет"
elif 30 <= age_input <= 34:
    Age = 3
    age_range = "30-34 лет"
elif 35 <= age_input <= 39:
    Age = 4
    age_range = "35-39 лет"
elif 40 <= age_input <= 44:
    Age = 5
    age_range = "40-44 лет"
elif 45 <= age_input <= 49:
    Age = 6
    age_range = "45-49 лет"
elif 50 <= age_input <= 54:
    Age = 7
    age_range = "50-54 лет"
elif 55 <= age_input <= 59:
    Age = 8
    age_range = "55-59 лет"
elif 60 <= age_input <= 64:
    Age = 9
    age_range = "60-64 лет"
elif 65 <= age_input <= 69:
    Age = 10
    age_range = "65-69 лет"
elif 70 <= age_input <= 74:
    Age = 11
    age_range = "70-74 лет"
elif 75 <= age_input <= 79:
    Age = 12
    age_range = "75-79 лет"
else:
    Age = 13
    age_range = "80+ лет"

# Отображение диапазона на сайте
st.write(f"Ваш возраст попадает в диапазон: {age_range}")

# Ввод роста (в см)
height = st.number_input("Введите ваш рост (в см)", min_value=50, max_value=250, value=170)

# Ввод веса (в кг)
weight = st.number_input("Введите ваш вес (в кг)", min_value=10, max_value=300, value=70)

# Преобразование роста и веса в индекс массы тела (BMI)
BMI = weight / ((height / 100) ** 2)

# Отображение рассчитанного BMI
st.write(f"Ваш индекс массы тела (BMI): {BMI:.2f}")

# --- Ввод признаков с "Да" / "Нет" ---

HighBP = st.selectbox("Есть ли повышенное давление?", ["Нет", "Да"])
HighBP = 1 if HighBP == "Да" else 0

HighChol = st.selectbox("Есть ли повышенный холестерин?", ["Нет", "Да"])
HighChol = 1 if HighChol == "Да" else 0

Smoker = st.selectbox("Курили ли вы более 100 сигарет за жизнь?", ["Нет", "Да"])
Smoker = 1 if Smoker == "Да" else 0

Stroke = st.selectbox("Был ли инсульт?", ["Нет", "Да"])
Stroke = 1 if Stroke == "Да" else 0

HeartDiseaseorAttack = st.selectbox("Были ли болезни сердца / инфаркт?", ["Нет", "Да"])
HeartDiseaseorAttack = 1 if HeartDiseaseorAttack == "Да" else 0

PhysActivity = st.selectbox("Была ли физическая активность за последние 30 дней?", ["Нет", "Да"])
PhysActivity = 1 if PhysActivity == "Да" else 0

Fruits = st.selectbox("Употребляете ли фрукты регулярно?", ["Нет", "Да"])
Fruits = 1 if Fruits == "Да" else 0

Veggies = st.selectbox("Употребляете ли овощи регулярно?", ["Нет", "Да"])
Veggies = 1 if Veggies == "Да" else 0

HvyAlcoholConsump = st.selectbox("Чрезмерное употребление алкоголя?", ["Нет", "Да"])
HvyAlcoholConsump = 1 if HvyAlcoholConsump == "Да" else 0
GenHlth = st.slider("Общее состояние здоровья (1 = отличное, 5 = плохое)", 1, 5, 3)
MentHlth = st.slider("Дней плохого психического состояния за 30 дней", 0, 30, 0)
PhysHlth = st.slider("Дней плохого физического состояния за 30 дней", 0, 30, 0)

DiffWalk = st.selectbox("Есть ли трудности при ходьбе?", ["Нет", "Да"])
DiffWalk = 1 if DiffWalk == "Да" else 0

Sex = st.selectbox("Пол (0 = женщина, 1 = мужчина)", ["Женщина", "Мужчина"])
Sex = 1 if Sex == "Мужчина" else 0

# --- Прогноз ---
if st.button("Узнать результат"):

    # ВАЖНО: порядок признаков должен совпадать с обучением модели
    features = np.array([[
        HighBP,
        HighChol,
        BMI,
        Smoker,
        Stroke,
        HeartDiseaseorAttack,
        PhysActivity,
        Fruits,
        Veggies,
        HvyAlcoholConsump,
        GenHlth,
        MentHlth,
        PhysHlth,
        DiffWalk,
        Sex,
        Age
    ]])

    probability = model.predict_proba(features)[0][1]

    # Порог можно менять для повышения Recall
    threshold = 0.37

    st.write(f"Вероятность риска диабета: {probability * 100:.2f}%")

    if probability >= threshold:
        st.error("⚠️ Повышенный риск диабета. Рекомендуется обратиться к врачу.")
    else:
        st.success("✅ Низкий риск диабета.")
