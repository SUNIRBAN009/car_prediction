from django.shortcuts import render
import os
from django.conf import settings
import pickle
import sklearn
import pandas as pd

model_path = os.path.join(settings.BASE_DIR, 'pkl/lrmodel.pkl')
encoder_path = os.path.join(settings.BASE_DIR, 'pkl/ordinalencoder.pkl')
scaler_path = os.path.join(settings.BASE_DIR, 'pkl/standardscaler.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)

with open(encoder_path, 'rb') as file:
    ordinal = pickle.load(file)

with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)

def home(request):
    prediction = 0
    print(model_path)
    print(encoder_path)
    print(scaler_path)
    
    if request.method == "POST":
        brand = request.POST.get('brand')
        car_model = request.POST.get('model')
        model_year = request.POST.get('model_year')
        mileage = request.POST.get('mileage')
        fuel_type = request.POST.get('fuel_type')
        transmission = request.POST.get('transmission')
        hp = request.POST.get('hp')
        engine_size = request.POST.get('engine_size')
        accident = request.POST.get('accident')
        clean_title = request.POST.get('clean_title')

        data = {
            "brand": brand,
            "model": car_model,
            "model_year": model_year,
            "milage": mileage,  # Fixed typo
            "fuel_type": fuel_type,
            "transmission": transmission,
            "accident": accident,
            "clean_title": clean_title,
            "hp": hp,
            "L": engine_size,  # Fixed case issue
        }


                # Check for missing values
        if None in data.values():
            print("Missing form values:", data)
            return render(request, "index.html", {"error": "Please fill all fields."})

        dataset = pd.DataFrame([data])
        print("Dataset Columns Before Encoding:", dataset.columns)
        categorical_data = ['brand', 'model', 'fuel_type', 'transmission', 'accident', 'clean_title']
        dataset[categorical_data] = ordinal.transform(dataset[categorical_data])
        dataset = scaler.transform(dataset)

        print("Processed Dataset:", dataset)
        prediction = model.predict(dataset)
        
    return render(request, "index.html", {'prediction':prediction})
