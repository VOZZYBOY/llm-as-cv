#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Analysis System for Forklift Assessment Model
Система анализа качества модели оценки погрузчиков
"""

import os
import json
import pandas as pd
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from gemini import ForkliftAssessment, analyze_forklift

class QualityAnalyzer:
    """Анализатор качества модели оценки погрузчиков"""
    
    def __init__(self, ground_truth_csv: str = "testCUP.csv"):
        self.ground_truth_csv = ground_truth_csv
        self.results = []
        self.metrics = {}
        
    def load_ground_truth(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.ground_truth_csv, encoding='utf-8')
            return df
            
            df.columns = ['id', 'Чистота', 'Окрашен без потертостей', 'зеркала', 'фары', 
                         'сиденье', 'крыша', 'стекло', 'Шины', 'Поломка', 'съемка']
            
            print(f"Загружено {len(df)} эталонных записей")
            return df
        except Exception as e:
            print(f"Ошибка загрузки CSV: {e}")
            return pd.DataFrame()
    
    def find_forklift_images(self, forklift_id: str) -> List[str]:
        """Поиск изображений конкретного погрузчика"""
        folder_path = Path(f"testCUP/{forklift_id}")
        if not folder_path.exists():
            return []
        
        images = list(folder_path.glob("*.jpg"))[:4]  
        return [str(img) for img in images]
    
    def compare_assessments(self, predicted: ForkliftAssessment, ground_truth: Dict) -> Dict:
        """
        Сравнение предсказания модели с эталоном
        
        Returns:
            Dict с результатами сравнения по каждому параметру
        """
        comparison = {}
        
        field_mapping = {
            'cleanliness': 'Чистота',
            'paint_condition': 'Окрашен_без_потертостей',
            'mirrors': 'Наличие_боковых_зеркал_или_1_панорамное',
            'lights': 'Передние_и_задние_фары_исправны_целы', 
            'seat': 'Целое_сидение',
            'roof': 'Целая_крыша_где_есть',
            'glass': 'Целое_стекло_где_есть',
            'tires': 'Шины_протектор_не_стерт',
            'other_breakdowns': 'другие_поломки',
            'photo_rules': 'правила_съемки'
        }
        
        for model_field, csv_field in field_mapping.items():
            predicted_value = getattr(predicted, model_field)
            raw_actual_value = ground_truth.get(csv_field, None)
            
            actual_value = self._convert_csv_value(raw_actual_value)
            
            if actual_value is not None:
                comparison[model_field] = {
                    'predicted': predicted_value,
                    'actual': actual_value,
                    'raw_actual': raw_actual_value,
                    'match': predicted_value == actual_value,
                    'diff': abs(predicted_value - actual_value) if actual_value != -1 else None
                }
        
        return comparison
    
    def _convert_csv_value(self, value) -> Optional[int]:
        """Преобразование значения из CSV в число"""
        if pd.isna(value) or value in ['нет', 'нет ']:
            return -1  # Отсутствует
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return None
    
    def calculate_metrics(self, comparisons: List[Dict]) -> Dict:
        """
        Расчет метрик качества
        
        Args:
            comparisons: Список результатов сравнений
            
        Returns:
            Словарь с метриками
        """
        metrics = {
            'overall_accuracy': 0,
            'field_accuracy': {},
            'error_analysis': {},
            'latency_metrics': {},
            'stability_metrics': {}
        }
        
        if not comparisons:
            return metrics
            
        # 1. Общая точность (Overall Accuracy)
        total_predictions = 0
        correct_predictions = 0
        
        field_stats = {}
        
        for comparison in comparisons:
            for field, result in comparison.items():
                if field not in field_stats:
                    field_stats[field] = {'correct': 0, 'total': 0, 'errors': []}
                
                if result['actual'] is not None and result['actual'] != -1:  # Исключаем N/A
                    field_stats[field]['total'] += 1
                    total_predictions += 1
                    
                    if result['correct']:
                        field_stats[field]['correct'] += 1
                        correct_predictions += 1
                    else:
                        field_stats[field]['errors'].append({
                            'predicted': result['predicted'],
                            'actual': result['actual'],
                            'diff': result['diff']
                        })
        
        # Общая точность
        metrics['overall_accuracy'] = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # 2. Точность по полям
        for field, stats in field_stats.items():
            metrics['field_accuracy'][field] = {
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'correct': stats['correct'],
                'total': stats['total'],
                'error_rate': (stats['total'] - stats['correct']) / stats['total'] if stats['total'] > 0 else 0
            }
        
        # 3. Анализ ошибок
        metrics['error_analysis'] = self._analyze_errors(field_stats)
        
        return metrics
    
    def _analyze_errors(self, field_stats: Dict) -> Dict:
        """Анализ типов ошибок"""
        error_analysis = {}
        
        for field, stats in field_stats.items():
            if not stats['errors']:
                continue
                
            errors = stats['errors']
            error_analysis[field] = {
                'total_errors': len(errors),
                'false_positives': sum(1 for e in errors if e['predicted'] > e['actual']),  # Модель завышает
                'false_negatives': sum(1 for e in errors if e['predicted'] < e['actual']),  # Модель занижает
                'avg_error_magnitude': np.mean([e['diff'] for e in errors if e['diff'] is not None])
            }
        
        return error_analysis
    
    def test_stability(self, forklift_id: str, num_runs: int = 3) -> Dict:
        """
        Тест стабильности результатов на одних и тех же изображениях
        
        Args:
            forklift_id: ID погрузчика для тестирования
            num_runs: Количество запусков
            
        Returns:
            Метрики стабильности
        """
        images = self.find_forklift_images(forklift_id)
        if not images:
            return {}
        
        results = []
        print(f"Тестирование стабильности на погрузчике {forklift_id} ({num_runs} запусков)")
        
        for run in range(num_runs):
            try:
                assessment = analyze_forklift(images)
                results.append(assessment.model_dump())
                print(f"Запуск {run + 1}/{num_runs} выполнен")
            except Exception as e:
                print(f"Запуск {run + 1}/{num_runs} ошибка: {e}")
        
        if len(results) < 2:
            return {"error": "Недостаточно данных для анализа стабильности"}
        
        # Анализ стабильности
        stability_metrics = {}
        
        for field in results[0].keys():
            values = [result[field] for result in results]
            stability_metrics[field] = {
                'variance': np.var(values),
                'std_dev': np.std(values),
                'min': min(values),
                'max': max(values),
                'range': max(values) - min(values),
                'all_same': len(set(values)) == 1
            }
        
        # Общая стабильность
        total_variations = sum(1 for field_stats in stability_metrics.values() if not field_stats['all_same'])
        stability_metrics['overall_stability'] = {
            'stable_fields': len(stability_metrics) - total_variations,
            'unstable_fields': total_variations,
            'stability_rate': (len(stability_metrics) - total_variations) / len(stability_metrics)
        }
        
        return stability_metrics
    
    def run_full_analysis(self, sample_size: Optional[int] = None) -> Dict:
        """
        Полный анализ качества модели
        
        Args:
            sample_size: Количество погрузчиков для анализа (None = все)
            
        Returns:
            Полный отчет о качестве
        """
        print("Запуск полного анализа качества модели")
        
        # Загрузка эталонных данных
        ground_truth_df = self.load_ground_truth()
        if ground_truth_df.empty:
            return {"error": "Не удалось загрузить эталонные данные"}
        
        # Ограничение выборки если нужно
        if sample_size:
            ground_truth_df = ground_truth_df.head(sample_size)
        
        comparisons = []
        failed_assessments = []
        latencies = []  # Добавляем список для времени выполнения
        
        print(f"Анализ {len(ground_truth_df)} погрузчиков")
        
        for idx, row in ground_truth_df.iterrows():
            forklift_id = str(int(float(row.iloc[0])))  # Преобразуем float в int, затем в string
            print(f"Анализирую погрузчик {forklift_id} ({idx + 1}/{len(ground_truth_df)})")
            
            # Поиск изображений
            images = self.find_forklift_images(forklift_id)
            if not images:
                print(f"Нет изображений для погрузчика {forklift_id}")
                continue
            
            try:
                # Измерение времени выполнения
                start_time = time.time()
                predicted = analyze_forklift(images)
                end_time = time.time()
                
                latency = end_time - start_time
                latencies.append(latency)
                print(f"Время анализа: {latency:.2f}с")
                
                # Сравнение с эталоном
                ground_truth_dict = row.to_dict()
                comparison = self.compare_assessments(predicted, ground_truth_dict)
                comparisons.append(comparison)
                
            except Exception as e:
                print(f"Ошибка анализа {forklift_id}: {e}")
                failed_assessments.append(forklift_id)
        
        # Расчет метрик
        metrics = self.calculate_metrics(comparisons)
        
        # Добавляем метрики латентности
        if latencies:
            metrics['latency_metrics'] = {
                'avg_latency': np.mean(latencies),
                'min_latency': np.min(latencies),
                'max_latency': np.max(latencies),
                'median_latency': np.median(latencies),
                'std_latency': np.std(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'total_samples': len(latencies)
            }
        
        # Добавление информации о провалах
        metrics['failed_assessments'] = failed_assessments
        metrics['success_rate'] = (len(comparisons)) / len(ground_truth_df)
        
        return metrics
    
    def generate_report(self, metrics: Dict, output_file: str = "quality_report.json"):
        """Генерация отчета о качестве"""
        
        # Сохранение в JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        # Консольный отчет
        print("\n" + "="*60)
        print("ОТЧЕТ О КАЧЕСТВЕ МОДЕЛИ")
        print("="*60)
        
        if 'overall_accuracy' in metrics:
            print(f"Общая точность: {metrics['overall_accuracy']:.2%}")
            print(f"Успешность анализа: {metrics.get('success_rate', 0):.2%}")
        
        if 'field_accuracy' in metrics:
            print("\nТочность по параметрам:")
            for field, stats in metrics['field_accuracy'].items():
                accuracy = stats['accuracy']
                status = "OK" if accuracy > 0.8 else "WARN" if accuracy > 0.6 else "FAIL"
                print(f"[{status}] {field}: {accuracy:.2%} ({stats['correct']}/{stats['total']})")
        
        if 'error_analysis' in metrics:
            print("\nАнализ ошибок:")
            for field, errors in metrics['error_analysis'].items():
                if errors['total_errors'] > 0:
                    fp_rate = errors['false_positives'] / errors['total_errors']
                    fn_rate = errors['false_negatives'] / errors['total_errors']
                    print(f"{field}: {errors['total_errors']} ошибок")
                    print(f"  Ложные срабатывания: {fp_rate:.1%}")
                    print(f"  Пропуски: {fn_rate:.1%}")
        
        if 'latency_metrics' in metrics:
            lat = metrics['latency_metrics']
            print("\nПроизводительность:")
            print(f"Среднее время: {lat['avg_latency']:.2f}с")
            print(f"Минимальное время: {lat['min_latency']:.2f}с") 
            print(f"Максимальное время: {lat['max_latency']:.2f}с")
            print(f"95-й перцентиль: {lat['p95_latency']:.2f}с")
            print(f"99-й перцентиль: {lat['p99_latency']:.2f}с")
            print(f"Стд. отклонение: {lat['std_latency']:.2f}с")
        
        print(f"\nПодробный отчет сохранен в: {output_file}")

def main():
    """Основная функция для запуска анализа"""
    analyzer = QualityAnalyzer()
    
    print("СИСТЕМА АНАЛИЗА КАЧЕСТВА ОЦЕНКИ ПОГРУЗЧИКОВ")
    print("="*55)
    print("1. Тест стабильности (одинаковые фото)")
    print("2. Анализ точности (сравнение с CSV)")
    print("3. Анализ конкретного погрузчика")
    print("4. Полный анализ")
    
    choice = input("Введите номер (1-4): ").strip()
    
    if choice == "1":
        forklift_id = input("Введите ID погрузчика (например, 136): ").strip()
        num_runs = int(input("Количество запусков (по умолчанию 3): ") or "3")
        
        stability_results = analyzer.test_stability(forklift_id, num_runs)
        print(json.dumps(stability_results, indent=2, ensure_ascii=False))
        
    elif choice == "2":
        sample_size = input("Количество погрузчиков для анализа (Enter = все): ").strip()
        sample_size = int(sample_size) if sample_size else None
        
        metrics = analyzer.run_full_analysis(sample_size)
        analyzer.generate_report(metrics)
        
    elif choice == "3":
        analyze_single_forklift(analyzer)
        
    elif choice == "4":
        # Сначала тест стабильности
        print("\nФаза 1: Тест стабильности")
        stability_results = analyzer.test_stability("136", 3)
        
        # Затем анализ точности
        print("\nФаза 2: Анализ точности")
        metrics = analyzer.run_full_analysis(5)  # Первые 5 погрузчиков
        
        # Комбинированный отчет
        combined_metrics = {
            "accuracy_metrics": metrics,
            "stability_metrics": stability_results
        }
        
        analyzer.generate_report(combined_metrics, "full_quality_report.json")
        
    else:
        print("Неверный выбор")

def analyze_single_forklift(analyzer: QualityAnalyzer):
    """Анализ качества для конкретного погрузчика"""
    forklift_id = input("Введите номер погрузчика (например, 1035, 136, 152): ").strip()
    
    # Проверяем наличие папки
    folder_path = Path(f"testCUP/{forklift_id}")
    if not folder_path.exists():
        print(f"Папка testCUP/{forklift_id} не найдена!")
        return
    
    # Загружаем эталонные данные
    try:
        ground_truth_df = analyzer.load_ground_truth()
        
        # Ищем погрузчик в CSV
        ground_truth_row = None
        for idx, row in ground_truth_df.iterrows():
            if str(row.iloc[0]) == forklift_id or row.iloc[0] == int(forklift_id):
                ground_truth_row = row
                break
        
        if ground_truth_row is None:
            print(f"Погрузчик {forklift_id} не найден в эталонных данных CSV")
            print("Выполняем анализ без сравнения с эталоном...")
            run_analysis_without_ground_truth(forklift_id)
            return
            
    except Exception as e:
        print(f"Ошибка загрузки CSV: {e}")
        print("Выполняем анализ без сравнения с эталоном...")
        run_analysis_without_ground_truth(forklift_id)
        return
    
    print(f"\nАнализ погрузчика {forklift_id}...")
    print("="*50)
    
    # Получаем фотографии
    photo_paths = get_forklift_photos_for_analysis(forklift_id)
    if not photo_paths:
        return
    
    # Анализируем
    start_time = time.time()
    try:
        assessment = analyze_forklift(photo_paths)
        end_time = time.time()
        
        # Выводим результат модели
        print("\nРезультат анализа модели:")
        print("-" * 30)
        display_assessment_results(assessment)
        
        # Сравниваем с эталоном
        ground_truth_dict = ground_truth_row.to_dict()
        comparison = analyzer.compare_assessments(assessment, ground_truth_dict)
        
        # Выводим эталонные данные
        print("\nЭталонные данные из CSV:")
        print("-" * 30)
        display_ground_truth(ground_truth_dict)
        
        # Выводим сравнение
        print("\nСравнение с эталоном:")
        print("-" * 30)
        display_comparison(comparison)
        
        # Рассчитываем метрики
        accuracy = calculate_single_accuracy(comparison)
        processing_time = end_time - start_time
        
        print(f"\nМетрики качества:")
        print("=" * 30)
        print(f"Общая точность: {accuracy:.1%}")
        print(f"Время обработки: {processing_time:.2f} сек")
        
        # Сохраняем результат
        result = {
            "forklift_id": forklift_id,
            "model_assessment": assessment.model_dump(),
            "ground_truth": {k: v for k, v in ground_truth_dict.items() if not pd.isna(v)},
            "comparison": comparison,
            "accuracy": accuracy,
            "processing_time": processing_time
        }
        
        output_file = f"single_analysis_{forklift_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Результат сохранен в файл: {output_file}")
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")

def get_forklift_photos_for_analysis(forklift_id: str) -> List[str]:
    """Получение путей к фотографиям для анализа"""
    folder_path = Path(f"testCUP/{forklift_id}")
    photo_paths = list(folder_path.glob("*.jpg"))[:4]
    
    print(f"Найдено {len(photo_paths)} фотографий:")
    for i, path in enumerate(photo_paths, 1):
        print(f"  {i}. {path.name}")
    
    return [str(path) for path in photo_paths]

def run_analysis_without_ground_truth(forklift_id: str):
    """Анализ без сравнения с эталоном"""
    photo_paths = get_forklift_photos_for_analysis(forklift_id)
    if not photo_paths:
        return
    
    start_time = time.time()
    try:
        assessment = analyze_forklift(photo_paths)
        end_time = time.time()
        
        print("\nРезультат анализа модели:")
        print("-" * 30)
        display_assessment_results(assessment)
        
        processing_time = end_time - start_time
        print(f"\nВремя обработки: {processing_time:.2f} сек")
        
        # Сохраняем результат
        result = {
            "forklift_id": forklift_id,
            "model_assessment": assessment.model_dump(),
            "processing_time": processing_time
        }
        
        output_file = f"single_analysis_{forklift_id}_no_ground_truth.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"Результат сохранен в файл: {output_file}")
        
    except Exception as e:
        print(f"Ошибка при анализе: {e}")

def display_assessment_results(assessment: ForkliftAssessment):
    """Отображение результатов оценки"""
    field_names = {
        'cleanliness': 'Чистота',
        'paint_condition': 'Окрашен без потертостей', 
        'mirrors': 'Зеркала',
        'lights': 'Фары',
        'seat': 'Сиденье',
        'roof': 'Крыша',
        'glass': 'Стекло',
        'tires': 'Шины',
        'other_breakdowns': 'Другие поломки',
        'photo_rules': 'Правила съемки'
    }
    
    for field, name in field_names.items():
        value = getattr(assessment, field)
        status = "Норма" if value == 1 else "Проблема" if value == 0 else "Отсутствует"
        print(f"  {name}: {status} ({value})")

def display_ground_truth(ground_truth: Dict):
    field_mapping = {
        'Чистота': 'Чистота',
        'Окрашен, без потертостей': 'Окрашен без потертостей',
        'Наличие боковых зеркал или 1 панорамное': 'Зеркала', 
        'Передние и задние фары исправны, целы': 'Фары',
        'Целое сидение': 'Сиденье',
        'Целая крыша, где есть': 'Крыша',
        'Целое стекло, где есть': 'Стекло',
        'Шины протектор не стерт': 'Шины',
        'другие поломки': 'Другие поломки',
        'правила сьемки': 'Правила съемки'
    }
    
    analyzer = QualityAnalyzer()
    for csv_field, display_name in field_mapping.items():
        if csv_field in ground_truth and not pd.isna(ground_truth[csv_field]):
            value = ground_truth[csv_field]
            converted = analyzer._convert_csv_value(value)
            status = "Норма" if converted == 1 else "Проблема" if converted == 0 else "Отсутствует"
            print(f"  {display_name}: {status} ({converted}, исходное: '{value}')")

def display_comparison(comparison: Dict):
    """Отображение результатов сравнения"""
    field_names = {
        'cleanliness': 'Чистота',
        'paint_condition': 'Окрашен без потертостей',
        'mirrors': 'Зеркала',
        'lights': 'Фары', 
        'seat': 'Сиденье',
        'roof': 'Крыша',
        'glass': 'Стекло',
        'tires': 'Шины',
        'other_breakdowns': 'Другие поломки',
        'photo_rules': 'Правила съемки'
    }
    
    for field, name in field_names.items():
        if field in comparison:
            comp = comparison[field]
            match_status = "СОВПАДЕНИЕ" if comp['match'] else "НЕСОВПАДЕНИЕ"
            print(f"  {name}: {match_status} (модель: {comp['predicted']}, эталон: {comp['actual']})")

def calculate_single_accuracy(comparison: Dict) -> float:
    """Расчет точности для одного погрузчика"""
    if not comparison:
        return 0.0
    
    matches = sum(1 for comp in comparison.values() if comp['match'])
    total = len(comparison)
    
    return matches / total if total > 0 else 0.0

if __name__ == "__main__":
    main()
