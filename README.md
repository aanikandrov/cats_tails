# Тестовое задание "Хвост кота"

Аникандров АМ

## 1. Подготовка данных

Фотографии брал отсюда: 
1) www.istockphoto.com 
2) https://ru.pinterest.com

Размеченный датасет (Roboflow):
https://app.roboflow.com/test-hfa5m/cats-2gvph/9

Папка data:
1) cats.v9i.coco-segmentation - размеченный датасет
2) cats_yolo - датасет с нормализованными координатами
3) cats_yolo_augmented - датасет с аугментациями (albumentations) (но при обучении итоговой модели не использовался, тк аугментация через параметры train показала себя лучше)
4) cats_another - несколько изображений котов которые не использовались при обучении

- utils/yolo_script.py - скрипт перевода COCO датасета в YOLO txt формат 
- utils/augment.py - создание версии датасета с аугментациями (albumentations)

## 2. Обучение

Использовалась модель YOLOv8n-seg

- core/train.py - запуск обучения модели
- core/valid.py - вывод метрики mAP50-95 

Веса итоговой модели лежат в models/final/best.pt

**Метрики итоговой модели**

P = 0.95
R = 0.66
mAP50 = 0.82
mAP50-95 = 0.58


## 3. Демонстрация

- demo/app_opencv.py - простой скрипт с использованием OpenCV GUI. Указываем путь к модели и к файлу - выводит изображение с маской.
- demo/app_graio.py - интерфейс Gradio.

### Примеры работы 
Несколько примеров predict на изображениях из папки data/cats_another

![image](https://github.com/user-attachments/assets/ff8ca2ee-f413-48bf-a299-7cb9765a632f)

!![image (1)](https://github.com/user-attachments/assets/9dfe5d7e-bbfc-4281-bd2b-f637c21bc216)

![image (2)](https://github.com/user-attachments/assets/4f916799-8a6e-4f2c-92a8-7bfc350d84a2)

