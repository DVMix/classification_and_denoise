# classification_and_denoise

Модели запакованы в универсальный для обеих задач пайплайн, который имеет 
методы для обучения б валидации и тестирования. 
В директории 'utils' находятся основные инструменты:
1. Архитектуры сетей (resnet.py + DnCNN)
2. Подгрузчик датасета (dataset.py)
3. Инструменты для подсчета средних показателей лосс-функций и ключевых метрик 
    (average_meter.py)
4. Основной класс-сборщик задачи (Experiment.py)
5. Вспомогательные инструменты для форматирования и предобработки данных
    (formatter.py)
    
Порядок действий:
1. Запуллить данный репозиторий
2. Распаковать датасет в репозиторий проекта в папку 'dataset'
3. Запустить обучение задач можно посредством баш-скриптов:
    task2_*_train.sh (вместо * 1 или 2, где 1 - классификацияб 2 - денойз)
    предварительно проинициализировав необходимые параметры
    
     - пространственные размерности входных в сеть тензоров
    size = 320
    
    - количество эпох обучения
    epochs = 11
    
    - инициализация архитектуры сети, размера батча
    full = True
    if full:
         batch_size= 15
         model     = resnet50(num_classes = 2)
         arch_name = 'ResNet50'
    else:
         batch_size= 30
         model     = resnet50_half_conv(num_classes = 2)
         arch_name = 'ResNetShort'
    - инициализация подгрузчиков данных 
    ds_train = Classification_DataSet(folder = 'dataset',split='train')
    ds_val   = Classification_DataSet(folder = 'dataset',split='val')
    
    - оптимизатор
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    
    - инициализация лосс-функции
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
    - инициалазация названия директории, где будут храниться веса обученных моделей и 
      файлы со статистиками
    output_dir   = 'checkpoints'
    
    - название задачи
    task_name    = 'Classification'
    
    - инициализация класса-сборщика статистики 
    stat_counter = classification_stat_counter()

    - инициализация основного класса-пайплайна для оперирования над моделями
    ex = Experimet(task_name = task_name,
                   arch_name = arch_name, 
                   net=model, 
                   epochs=epochs,
                   train_set=ds_train, 
                   val_set=ds_val,
                   formatter=formatter_classification, 
                   optimizer=optimizer,
                   loss_fn=loss_fn,
                   stat_counters = stat_counter, 
                   output_dir=output_dir,
                   img_size=size,
                   batch_size=batch_size)
                   
    - вызов метода для начала обучения модели
    ex.run()

4. Для оценки полученных результатов - воспользоваться методом класса 
    ex.validate()
    
5. Метод класса infer  принимает тестовые данные (путь к спектрограммам или сами спектрограммы) + аннотации
   В случае задачи классификации:
    - выдается тип спектрограммы (чистая или зашумленная)
    - если аннотации не пустое значение - результат предсказания (верно или неверно)
    
   В случае задачи денойза:
    - если аннотации не пустое значение - MSE для чистой и зашумленной спектрограммы
    
6. Классификация: модель - базовая resnet50 (с предобработкой взодных тензоров под необходимый формат)
   Денойз: полносверточная сеть, кол-во слоев настраиваемо.
   
7. В ходе обучения чекпойнты сохраняются в корне проекта в папке checkpoints/{task_name}/{net_name}/{prepoc_tensor_size}
    В директории: чекпойнты, json-файл с статистиками, txt-файл с названием свежего чекпойнта