
# Тестовое задание "Хвост кота"

Аникандров АМ

## 1. Подготовка данных

Фотографии брал отсюда: 
1) www.istockphoto.com 
2) https://ru.pinterest.com

Размеченный датасет (Roboflow):
https://app.roboflow.com/test-hfa5m/cats-2gvph/9

Папки в проекте:
1) cats.v9i.coco-segmentation - размеченный датасет
2) cats_yolo - датасет с нормализованными координатами
3) cats_yolo_augmented - датасет с аугментациями (albumentations) (но при обучении итоговой модели не использовался, тк аугментация через параметры train показала себя лучше)
4) cats_another - несколько изображений котов которые не использовались при обучении

- scripts/utils/yolo_script.py - скрипт перевода COCO датасета в YOLO txt формат 
- scripts/utils/augment.py - создание версии датасета с аугментациями (albumentations)

## 2. Обучение

Использовалась модель YOLOv8n-seg

- scripts/core/train.py - запуск обучения модели
- scripts/core/valid.py - вывод метрики mAP50-95 

Веса итоговой модели лежат в models/final/best.pt

**Метрики итоговой модели**

P = 0.95
R = 0.66
mAP50 = 0.82
mAP50-95 = 0.58


## 3. Демонстрация

- ui/app_opencv.py - простой скрипт с использованием OpenCV GUI. Указываем путь к модели и к файлу - выводит изображение с маской.
- ui/app_graio.py - интерфейс на Gradio.
