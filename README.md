# Тестовое задание "Хвост кота"

Прототип системы машинного зрения для сегментации хвоста кота на изображениях.  

Выполнил: Аникандров Андрей Максимович


## 1. Подготовка данных

Фотографии брал отсюда: 
  1) www.istockphoto.com 
  2) https://ru.pinterest.com

Размеченный датасет (Roboflow):
https://app.roboflow.com/test-hfa5m/cats-2gvph/9

Папка data:
  - cats.v9i.coco-segmentation - размеченный датасет
  - cats_yolo - датасет с нормализованными координатами
  - cats_yolo_augmented - датасет с аугментациями (albumentations) (но при обучении итоговой модели не использовался, тк аугментация через параметры train показала себя лучше)
  - cats_another - несколько изображений котов которые не использовались при обучении

Файлы для подготовки данных:
  - utils/yolo_script.py - скрипт перевода COCO датасета в YOLO txt формат 
  - utils/augment.py - создание версии датасета с аугментациями (albumentations)


## 2. Обучение

Использовалась модель YOLOv8n-seg. 
Веса итоговой обученной модели лежат в models/final/best.pt. 

Файлы:
  - core/train.py - запуск обучения модели
  - core/valid.py - вывод метрики mAP50-95
  - core/predict.py - вызов predict

Подобранные гиперпараметры:
  - epochs = 120
  - batch = 16
  - dropout = 0.3
  - freeze = [0..10]

Аугментации:
  - hsv_h=0.015
  - hsv_s=0.7
  - hsv_v=0.4
  - degrees=30
  - translate=0.2
  - scale=0.5
  - shear=10
  - fliplr=0.5

**Метрики итоговой модели**
  - P = 0.95
  - R = 0.66
  - mAP50 = 0.82
  - mAP50-95 = 0.58


## 3. Демонстрация

- demo/app_opencv.py - простой скрипт с использованием OpenCV GUI. Указываем путь к модели и к изображению - выводит изображение с маской
- demo/app_graio.py - интерфейс Gradio для изображений

### Примеры работы 
Несколько примеров predict на изображениях из папки data/cats_another

![image](https://github.com/user-attachments/assets/ff8ca2ee-f413-48bf-a299-7cb9765a632f)

!![image (1)](https://github.com/user-attachments/assets/9dfe5d7e-bbfc-4281-bd2b-f637c21bc216)

![image (2)](https://github.com/user-attachments/assets/4f916799-8a6e-4f2c-92a8-7bfc350d84a2)

