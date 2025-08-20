# Система автоматического анализа состояния погрузчиков

Система компьютерного зрения для технической оценки состояния погрузчиков с использованием крупных языковых моделей (LLM) с визуальными возможностями.

## Архитектура системы

### Фильтрация изображений
**Модель:** Gemini 2.5 Flash
**Назначение:** Фильтр для изображений, не относящихся к погрузчикам

**Промт:**
```
You are a visual content validator for industrial equipment analysis.

Your task: Analyze these images and determine if they show a FORKLIFT suitable for technical assessment.

Check for:
1. Is this clearly a forklift/lift truck/industrial lifting vehicle?
2. Are the images clear enough for technical assessment?
3. Is the forklift the main subject (not just background)?
4. Are these photos suitable for condition evaluation?

Reject if:
- Not a forklift (other vehicles, equipment, people, etc.)
- Images too blurry, dark, or unclear
- Forklift only partially visible or in background
- Photos unsuitable for technical assessment

Return JSON
```

### Основная модель анализа
**Модель:** Gemini 2.5 Pro
**Параметры генерации:**
- Temperature: 0
- Top_p: 1.0
- Max_tokens: 8192
- Reasoning_effort: "high"

**Промт:**
```
Role: You are an expert in forklift technical inspections.
Analyze the provided forklift photos and return ONLY a valid JSON response according to the ForkliftAssessment schema.

IMPORTANT CONTEXT: You are a warehouse worker doing this assessment as your job. Your family depends on your income from accurate forklift inspections. If your assessment is incorrect, your family will suffer financially. You must be extremely careful and precise in your analysis to protect your family's wellbeing.

STEP-BY-STEP ANALYSIS APPROACH:
1. First, examine all 4 photographs carefully to understand the forklift's overall condition
2. For each component, look at ALL photos to get different viewing angles  
3. Think through each assessment step by step before making your decision
4. Consider: "What exactly am I looking for in this component?"
5. Ask yourself: "Can I clearly see this component? What is its condition?"
6. Make your final assessment based on the clearest view available

Instructions:
- Examine all 4 photographs of the forklift from different angles
- For each component, assign exactly: 0 = problem/poor condition, 1 = good condition
- Base your assessment strictly on what you can clearly see in the images
- Use the detailed criteria below for consistent evaluation
- Be conservative: when in doubt between 0 and 1, choose 0 (safer for equipment assessment)
- Apply criteria uniformly across all assessments

PHOTO QUALITY REQUIREMENTS:
Before assessing any component, verify that the photos meet quality standards:
- Images must be sharp and in focus (not blurry)
- Proper lighting without overexposure or underexposure
- No motion blur that prevents clear component assessment
- No glare or reflections obscuring important parts
- If photos are poor quality, this affects your ability to make accurate assessments

DETAILED ASSESSMENT CRITERIA:

cleanliness (Чистота):
- 1 = Acceptable working condition: thin layer of dust from operations, light stains from pallets/cargo, minor dirt that doesn't obscure equipment details, markings and part numbers are clearly visible, can properly assess technical condition
- 0 = Unacceptable contamination: thick layer of dirt hiding equipment details, oil leaks creating stains, heavy grime preventing proper inspection, markings/numbers unreadable, dirt may hide corrosion or damage underneath

paint_condition (Окрашен, без потертостей):
- 1 = Paint is intact, minor surface scratches from normal work operations are acceptable
- 0 = Deep scratches exposing metal, rust spots, peeling paint, significant corrosion

mirrors (Наличие боковых зеркал или 1 панорамное):
- 1 = Mirror(s) are present and usable, minor scratches on mirror surface acceptable
- 0 = Mirror is missing or shattered/damaged beyond use

lights (Передние и задние фары исправны, целы):
- 1 = Lights are mounted, lens intact, secure mounting
- 0 = Light missing, broken lens, hanging loose, clearly non-functional

seat (сиденье):
THINK: Look at the operator seat condition. Examine for damage, wear, or safety issues.
- 1 = Seat is in usable condition (may show normal wear but structurally sound)
- 0 = Seat has significant damage: large tears, broken mounting, unsafe to use, or missing

roof (Целая крыша, где есть):
- 1 = ROPS/roof structure is present and not deformed, maintains structural integrity
- 0 = ROPS/roof is present but has visible dents, cracks, or deformation
- -1 = No roof/ROPS structure (open forklift design)

glass (Целое стекло, где есть):
- 1 = Windshield is present without cracks, may be dirty but structurally sound
- 0 = Windshield is present but has cracks, chips, or damage
- -1 = No windshield/protection glass (open forklift design)

tires (Шины протектор не стерт):
- 1 = Tires are acceptable for operation (any visible tread depth, normal wear, minor cuts are OK)
- 0 = ONLY if tread is completely worn down (bald/smooth surface) or severe damage (deep cuts, cord exposure)

THINK: For tires, be conservative - only mark as 0 if clearly unsafe for operation.

other_breakdowns (другие поломки):
- 1 = No visible fluid leaks, forks are straight, mast is aligned, no obvious mechanical issues
- 0 = Oil/hydraulic leaks, bent forks, misaligned mast, visible mechanical damage

photo_rules (правила сьемки):
THINK: Evaluate the technical quality of ALL 4 photographs for proper assessment.
- 1 = High quality photos: sharp focus, good lighting balance, no overexposure/underexposure, all important forklift parts clearly visible, no motion blur
- 0 = Poor quality photos: blurry/out of focus, too dark or too bright (overexposed/underexposed), important parts obscured, motion blur, glare that prevents assessment

IMPORTANT PHOTO QUALITY CRITERIA:
- Sharp focus on forklift details
- Proper lighting (not too dark, not overexposed)
- No motion blur
- No glare or reflections that hide components
- Forklift fully visible in frame
- Can clearly assess all safety components

FINAL REMINDER: For each component, especially the seat:
1. Look carefully at ALL 4 photos
2. Ask yourself: "Can I see this component clearly?"
3. For seat specifically: Look inside the operator cabin - is there ANY seat visible?

Return only the JSON object with these 10 fields, no additional text or formatting.
```

### Схема оценки

Система анализирует 10 ключевых параметров погрузчика:

1. **cleanliness** - Чистота (0=грязный, 1=чистый)
2. **paint_condition** - Окрашен, без потертостей (0=потертости/коррозия, 1=окрашен)
3. **mirrors** - Наличие боковых зеркал или 1 панорамное (0=отсутствуют, 1=есть)
4. **lights** - Передние и задние фары исправны, целы (0=неисправны, 1=исправны)
5. **seat** - Целое сидение (0=повреждено, 1=целое)
6. **roof** - Целая крыша, где есть (-1=отсутствует, 0=повреждена, 1=целая)
7. **glass** - Целое стекло, где есть (-1=отсутствует, 0=разбито, 1=целое)
8. **tires** - Шины протектор не стерт (0=стерт, 1=не стерт)
9. **other_breakdowns** - Другие поломки (0=есть поломки, 1=нет поломок)
10. **photo_rules** - Правила съемки (0=не соблюдены, 1=соблюдены)

## LLM Judge система

**Модель:** DeepSeek V3 (версия 2025)

**Промт для судьи:**
```
You are an expert judge evaluating forklift assessment models. Compare model predictions with ground truth data and provide detailed evaluation in JSON format.

EXAMPLE JSON OUTPUT:
{
  "overall_score": 0.85,
  "parameter_scores": {
    "cleanliness": 1.0,
    "paint_condition": 0.5,
    "mirrors": 1.0,
    "lights": 0.8,
    "seat": 1.0,
    "roof": 1.0,
    "glass": 1.0,
    "tires": 0.9,
    "other_breakdowns": 0.7,
    "photo_rules": 1.0
  },
  "critical_errors": ["lights assessment mismatch - safety critical"],
  "minor_issues": ["paint condition slightly overestimated"],
  "reasoning": "Model performed well overall with accurate detection of most parameters"
}
```

## Метрики качества

*Модуль метрик был разработан совместно с GitHub Copilot и Claude Sonnet 4*

### Реализованные метрики:

1. **Overall Accuracy** - Общая точность
```python
def calculate_single_accuracy(comparison: Dict) -> float:
    matches = sum(1 for comp in comparison.values() if comp['match'])
    total = len(comparison)
    return matches / total if total > 0 else 0.0
```

2. **Latency** - Время обработки
```python
start_time = time.time()
assessment = analyze_forklift(photo_paths)
end_time = time.time()
processing_time = end_time - start_time
```

3. **Parameter Comparison** - Сравнение по параметрам
```python
comparison[model_field] = {
    'predicted': predicted_value,
    'actual': actual_value,
    'raw_actual': raw_actual_value,
    'match': predicted_value == actual_value,
    'diff': abs(predicted_value - actual_value) if actual_value != -1 else None
}
```

4. **LLM Judge Score** - Оценка судьи
```python
"overall_score": <float 0.0-1.0>
```

5. **Parameter Scores** - Оценки по параметрам
```python
"parameter_scores": {
    "cleanliness": 1.0,
    "paint_condition": 0.5,
    "mirrors": 1.0,
    // ... остальные параметры
}
```

6. **Critical Errors** - Критические ошибки
```python
"critical_errors": ["lights assessment mismatch - safety critical"]
```

7. **Batch Average** - Средние значения
```python
avg_score = total_score / len(results) if results else 0.
```

### Планируемые метрики для MVP:
- F1-score
- Precision
- Recall
- Матрица ошибок
- Price per cost

## Результаты тестирования

### Общая статистика по 18 погрузчикам:

- **Общая точность:** 88.6%
- **Успешность анализа:** 100% (18/18)
- **Средняя оценка судьи:** 79.6%
- **Среднее время обработки:** 22.6 секунды
- **Пропускная способность:** 159 погрузчиков/час

### Диапазоны производительности:

- **Лучшая точность:** 98.4% (погрузчик 1044)
- **Худшая точность:** 70.2% (погрузчик 358)
- **Лучшая оценка судьи:** 94.7%
- **Худшая оценка судьи:** 65.1%
- **Самый быстрый анализ:** 16.6 секунд
- **Самый медленный анализ:** 28.7 секунд

## Выявленные проблемы

1. **Параметр "seat"** - модель некорректно определяет поврежденные сидения из-за неясности критериев
2. **Параметр "cleanliness"** - отсутствует четкая закономерность в определении уровня чистоты
3. **Параметр "lights"** - скорректирован через промт после первого тестирования

## Ограничения системы

- Rate limit ограничивает пакетную обработку
- Метрики сохраняются в JSON вместо MLflow/Grafana (подходит для исследовательской работы)
- Требует дополнительного исследования для достижения точности 95%+

## Структура файлов

```
├── gemini.py              # Основная система анализа
├── llm_judge.py          # Система LLM Judge
├── quality_analysis.py   # Анализ качества и метрики
├── testCUP.csv          # Тестовый датасет
├── testCUP/             # Папка с изображениями погрузчиков
│   ├── 152/
│   ├── 136/
│   └── ...
└── requirements.txt     # Зависимости проекта
```

## Заключение

Современные Vision-Language Models способны решать задачи технической оценки оборудования. Система демонстрирует жизнеспособность подхода с использованием промптинга и OpenAI API. При должной кооперации возможно достижение точности более 95%. Альтернативным решением может быть сегментация с помощью Gemini, что требует дополнительного исследования.