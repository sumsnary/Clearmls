from clearml import Task
import functools
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_csv_data(file_path):
    """Загружает данные из CSV-файла."""
    try:
        logging.info(f"Загрузка данных из файла: {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"Данные успешно загружены. Количество строк: {len(data)}")
        return data
    except FileNotFoundError:
        logging.error(f"Ошибка: Файл не найден: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise


def clearml_experiment(project, experiment_name=None, tags=None):
    """Декоратор для отслеживания экспериментов с помощью ClearML."""

    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            
            
            task_name = experiment_name or function.__name__
            
            logging.info(f"Инициализация задачи ClearML: project='{project}', task='{task_name}', tags={tags}")
            task = Task.init(project_name=project, task_name=task_name, tags=tags or [])

            # Логирование параметров
            all_args = {f"arg_{i}": arg for i, arg in enumerate(args)}
            all_args.update(kwargs)
            
            logging.info(f"Параметры эксперимента: {all_args}")
            
            task.connect(all_args)
            
            try:
                logging.info("Запуск основной логики...")
                result = function(*args, **kwargs)
                logging.info("Основная логика успешно выполнена.")

                task.get_logger().report_scalar(
                    title="Execution", series="Status", value=1, iteration=0
                )
                logging.info("Отчет о статусе выполнения: Success.")
                
                
                if isinstance(result, pd.DataFrame):
                    
                    task.upload_artifact(name="processed_data", artifact_object=result)
                    logging.info("Данные сохранены как артефакт")

                    fig, ax = plt.subplots()
                    
                    if 'score' in result.columns:
                        ax.plot(result['score'])
                        task.get_logger().report_matplotlib_figure(
                        title="Data Plot", series="Dataset", figure=fig
                        )
                        logging.info("График построен и загружен.")
                    else:
                        logging.warning('Колонка score отсутствует, график не будет построен')
                
                return result
            
            except Exception as e:
                logging.error(f"Ошибка при выполнении эксперимента: {e}")
                task.get_logger().report_scalar(
                   title="Execution", series="Status", value=0, iteration=0
                )
                task.get_logger().report_text(f"Error details: {str(e)}")
                raise

        return wrapper

    return decorator


@clearml_experiment(
    project="ML_Lab_Experiments", experiment_name="Data_Cleaning", tags=["preprocessing", "v2"]
)
def clean_data(file_path, lower_threshold=0.2, upper_threshold=0.8, *args, **kwargs):
    """Очищает данные на основе заданных пороговых значений."""
    data = load_csv_data(file_path)
    
    logging.info(f"Начало отсечения выбросов. Нижний порог: {lower_threshold}, Верхний порог: {upper_threshold}")
    
    if 'score' in data.columns:
        cleaned_data = data[(data['score'] >= lower_threshold) & (data['score'] <= upper_threshold)]
        logging.info(f"Удалено строк: {len(data) - len(cleaned_data)}")
    else:
       cleaned_data = data
       logging.warning('Колонка score отсутствует, отсечение выбросов не произведено')
    
    
    return cleaned_data


# Использование
file_path = r"C:\Users\Lord\Desktop\venvbarf\venv\dataset.csv"
result = clean_data(file_path, lower_threshold=0.3, upper_threshold=0.9)

print('Обработанные данные:')
print(result)
