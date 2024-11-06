import gradio as gr
import joblib
import pandas as pd
import numpy as np

# Загрузка моделей
catboost = joblib.load('/home/denrann/anaconda3/envs/chat_apart_bot/Project/model/catboost/catb_model.pkl')
catboostWCF = joblib.load('/home/denrann/anaconda3/envs/chat_apart_bot/Project/model/catboost/catb_with_cat_features_model.pkl')
xgboost = joblib.load('/home/denrann/anaconda3/envs/chat_apart_bot/Project/model/xgboost/xgb_model.pkl')
knnreg = joblib.load('/home/denrann/anaconda3/envs/chat_apart_bot/Project/model/knn/knn_model.pkl')

# Загрузка моделей инкодеров
scaler = joblib.load('/home/denrann/anaconda3/envs/chat_apart_bot/Project/model/encoders/scaler.pkl')
target_fe_enc = joblib.load('/home/denrann/anaconda3/envs/chat_apart_bot/Project/model/encoders/frequency_encoder.pkl')


# Функция для нормализации входных признаков и прогнозирования стоимости жилья
def predict_house_price(model_choice, apartment_type, metro_station,    
   minutes_to_metro, region, number_of_rooms, 
   area, living_area, kitchen_area, floor, 
   number_of_floors, renovation):
    global catboost  # Объявляем catboost как глобальную переменную
    global catboostWCF  # Объявляем catboostWCF как глобальную переменную
    global xgboost  # Объявляем xgboost как глобальную переменную    
    global knnreg  # Объявляем knnreg как глобальную переменную 
 
    if model_choice == 'catboost':
        model = catboost
    elif model_choice == 'xgboost':
        model = xgboost
    elif model_choice == 'knnreg':
        model = knnreg
    elif model_choice == 'catboostWCF':
        model = catboostWCF 
    else:
        return "Invalid model choice"
    
    # Создание DataFrame с введенными признаками
    input_features = pd.DataFrame({
        'apartment_type': [apartment_type],
        'metro_station': [metro_station],
        'minutes_to_metro': [minutes_to_metro],
        'region': [region],
        'number_of_rooms': [number_of_rooms],
        'area': [area],
        'living_area': [living_area],
        'kitchen_area': [kitchen_area],
        'floor': [floor],
        'number_of_floors': [number_of_floors],
        'renovation': [renovation]
    })
 
    if model_choice == 'catboostWCF':
        # Предсказание стоимости жилья для модели с категориальными признаками
        predicted_price = model.predict(input_features)[0]
        return int(np.exp(predicted_price))
    else:
        # Отбор категориальных признаков
        features_cat = input_features.select_dtypes(include='object').columns
        
        # Преобразование категориальных признаков с помощью frequency encoding
        input_features[features_cat] = target_fe_enc.transform(input_features[features_cat])
        
        # Нормализация данных с помощью scaler
        normalized_features = scaler.transform(input_features)
        
        # Предсказание стоимости жилья
        predicted_price = model.predict(normalized_features)[0]
        return int(np.exp(predicted_price))

# Создание чат-бот интерфейса
with open('/home/denrann/anaconda3/envs/chat_apart_bot/Project/data/metro_station.txt', 'r', encoding='utf-8') as file:
    metro_stations_str = file.read()
    metro_stations = [station.strip() for station in metro_stations_str.split(',')]
      
# Создание интерфейса

chatbot_interface = gr.Interface(
    fn=predict_house_price,
    inputs=[
	    gr.Dropdown(label="Choose Model", choices=['catboost', 'xgboost', 'knnreg', 'catboostWCF']),
        gr.Dropdown(label="Тип жилья", choices=['Secondary', 'New building']),
        gr.Dropdown(label="Станция метро или ж/д", choices=metro_stations),
        gr.Number(label="Минут до метро"),
        gr.Dropdown(label="Регион", choices=['Moscow region', 'Moscow']),
        gr.Number(label="Количество комнат"),
        gr.Number(label="Площадь (кв.м)"),
        gr.Number(label="Жилая площадь (кв.м)"),
        gr.Number(label="Площадь кухни (кв.м)"),
        gr.Number(label="Этажей в доме"),
        gr.Number(label="Номер этажа"),
        gr.Dropdown(label="Ремонт", choices=['Cosmetic', 'European-style renovation', 'Without renovation', 'Designer'])
    ],
    
    outputs=gr.Textbox(label="Ориентировочная стоимость в рублях"),
    title='Определение стоимости жилья',
    description='Выберите значения для получения результата',
    allow_flagging='never',  # Установка параметра flagging в False
    submit_btn='Расчет',
    clear_btn='Очистка'
)

# Запуск чат-бот интерфейса
if __name__ == "__main__":
    chatbot_interface.launch()
