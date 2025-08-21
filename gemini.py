
import os
import json
import base64
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

class ForkliftAssessment(BaseModel):
    cleanliness: int = Field(description="Чистота: 0=грязный, 1=чистый") 
    paint_condition: int = Field(description="Окрашен, без потертостей: 0=потертости/коррозия, 1=окрашен")
    mirrors: int = Field(description="Наличие боковых зеркал или 1 панорамное: 0=отсутствуют, 1=есть")
    lights: int = Field(description="Передние и задние фары исправны, целы: 0=неисправны, 1=исправны")
    seat: int = Field(description="Целое сидение: 0=повреждено, 1=целое")
    roof: int = Field(description="Целая крыша, где есть: -1=отсутствует, 0=повреждена, 1=целая")
    glass: int = Field(description="Целое стекло, где есть: -1=отсутствует, 0=разбито, 1=целое")
    tires: int = Field(description="Шины протектор не стерт: 0=стерт, 1=не стерт")
    other_breakdowns: int = Field(description="другие поломки: 0=есть поломки, 1=нет поломок")
    photo_rules: int = Field(description="правила сьемки: 0=не соблюдены, 1=соблюдены")

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Валидность изображений для анализа погрузчика")
    confidence: float = Field(description="Уверенность в оценке от 0.0 до 1.0")

API_KEY = "AIzaSyD-uk8r54wjpIeOB7eIyZPubpHpsv3oeKg"
MODEL_NAME = "gemini-2.5-pro"
VALIDATION_MODEL = "gemini-2.5-flash-lite"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

ANALYSIS_PROMPT = """
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
IMPORTANT: Lights must be CLEAN and INTACT for safe operation
- 1 = Lights are mounted, lenses intact, NOT completely covered with dirt, light can pass through
- 0 = Light is missing, broken, or COMPLETELY covered with dirt (light cannot pass through)

seat (сиденье):
IMPORTANT: Seat must be PRESENT and WITHOUT SERIOUS DAMAGE
- 1 = Seat EXISTS and is usable (minor wear acceptable, main thing - not torn and not broken)
- 0 = Seat is MISSING, has large holes/tears, or broken mounting

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

photo_rules (Photo Quality Assessment):
THINK: Evaluate the technical quality of ALL 4 photographs for assessment capability.
- 1 = Acceptable photo quality: Images may have minor blur but details are clearly visible, adequate resolution to see forklift components, lighting allows proper assessment, can distinguish between parts and their condition
- 0 = Unacceptable photo quality: ONLY if severely blurred throughout, extremely poor resolution making it difficult to identify what's happening, cannot distinguish forklift details due to image quality issues, lighting so poor that assessment is impossible

IMPORTANT: Only mark as 0 if photo quality genuinely prevents technical assessment of the forklift condition.

FINAL REMINDER: For each component, especially the seat:
1. Look carefully at ALL 4 photos
2. Ask yourself: "Can I see this component clearly?"
3. For seat specifically: Look inside the operator cabin - is there ANY seat visible?

Return only the JSON object with these 10 fields, no additional text or formatting.

"""


def load_image_as_base64(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def prepare_images_for_openai(image_paths: List[str]) -> List[dict]:
    images = []
    for path in image_paths:
        if not os.path.exists(path):
            continue

        ext = Path(path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            mime_type = "image/jpeg"
        elif ext == '.png':
            mime_type = "image/png"
        else:
            continue

        b64_data = load_image_as_base64(path)

        image_obj = {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64_data}"
            }
        }

        images.append(image_obj)

    return images

def validate_forklift_images(image_paths: List[str]) -> ValidationResult:
    images = prepare_images_for_openai(image_paths)
    
    validation_prompt = """Analyze these images to determine if they show a FORKLIFT suitable for technical assessment.

Check:
1. Is this clearly a forklift/lift truck/industrial lifting vehicle?
2. Are images clear enough for technical assessment?
3. Is the forklift the main subject?
4. Are photos suitable for condition evaluation?

Set is_valid to true only if all criteria are met.
Set confidence based on image quality and clarity (0.0 to 1.0)."""
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": validation_prompt}
            ] + images
        }
    ]
    
    response = client.beta.chat.completions.parse(
        model=VALIDATION_MODEL,
        messages=messages,
        response_format=ValidationResult,
        temperature=0.1,
        max_tokens=100
    )

    usage = response.usage
    forklift_id = Path(image_paths[0]).parent.name if image_paths else None
    rec = {
        "model": VALIDATION_MODEL,
        "stage": "validation", 
        "forklift_id": forklift_id,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }
    with open("usage_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    result = response.choices[0].message.parsed
    if result is None:
        return ValidationResult(is_valid=True, confidence=0.5)
    
    return result

def analyze_forklift(image_paths: List[str]) -> ForkliftAssessment:
    validation_result = validate_forklift_images(image_paths)
    
    if not validation_result.is_valid:
        return ForkliftAssessment(
            cleanliness=0,
            paint_condition=0, 
            mirrors=0,
            lights=0,
            seat=0,
            roof=0,
            glass=0,
            tires=0,
            other_breakdowns=0,
            photo_rules=0
        )
    
    images = prepare_images_for_openai(image_paths)

    messages = [
        {
            "role": "system",
            "content": "You are an expert in forklift technical inspections. Analyze photos and return strictly structured JSON."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": ANALYSIS_PROMPT}
            ] + images
        }
    ]

    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        response_format=ForkliftAssessment,
        temperature=0,
        top_p=1.0,
        max_tokens=8192,
        reasoning_effort="high"
    )

    usage = response.usage
    forklift_id = Path(image_paths[0]).parent.name if image_paths else None
    rec = {
        "model": MODEL_NAME,
        "stage": "analysis",
        "forklift_id": forklift_id,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens
    }
    with open("usage_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    result = response.choices[0].message.parsed
    if result is None:
        return ForkliftAssessment(
            cleanliness=0, paint_condition=0, mirrors=0, lights=0, seat=0,
            roof=0, glass=0, tires=0, other_breakdowns=0, photo_rules=0
        )
    
    return result

def display_results(assessment: ForkliftAssessment):
    fields = {
        'cleanliness': 'Чистота',
        'paint_condition': 'Окрашен, без потертостей',
        'mirrors': 'Наличие боковых зеркал или 1 панорамное',
        'lights': 'Передние и задние фары исправны, целы',
        'seat': 'Целое сидение',
        'roof': 'Целая крыша, где есть',
        'glass': 'Целое стекло, где есть',
        'tires': 'Шины протектор не стерт',
        'other_breakdowns': 'другие поломки',  
        'photo_rules': 'правила сьемки'  
    }

    results = {}
    for field, label in fields.items():
        value = getattr(assessment, field)
        if value == 1:
            status = "Норма"
        elif value == 0:
            status = "Проблема"
        elif value == -1:
            status = "Отсутствует"
        else:
            status = "Неизвестно"
        results[label] = status
    return results

def get_forklift_photos(forklift_id: str) -> List[str]:
    folder_path = Path(f"testCUP/{forklift_id}")
    
    if not folder_path.exists():
        return []
    
    photo_paths = list(folder_path.glob("*.jpg"))
    
    if not photo_paths:
        return []
    
    photo_paths = photo_paths[:4]
    
    return [str(path) for path in photo_paths]

def main():
    choice = input("Выберите режим (1-4): ").strip()
    
    if choice == "1":
        forklift_id = input("Введите номер папки погрузчика: ").strip()
        
        photo_paths = get_forklift_photos(forklift_id)
        if not photo_paths:
            return
        
        assessment = analyze_forklift(photo_paths)
        results = display_results(assessment)
        
        for label, status in results.items():
            print(f"   {label}: {status}")
        
        output_file = f"forklift_{forklift_id}_assessment.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(assessment.model_dump(), f, ensure_ascii=False, indent=2)
        
    elif choice == "2":
        photo_paths = [
            "testCUP/152/photo_2025-08-07_16-47-20.jpg",
            "testCUP/152/photo_2025-08-07_16-47-22.jpg", 
            "testCUP/152/photo_2025-08-07_16-47-25.jpg",
            "testCUP/152/photo_2025-08-07_16-47-27.jpg"
        ]
        
        assessment = analyze_forklift(photo_paths)
        results = display_results(assessment)
        
        for label, status in results.items():
            print(f"   {label}: {status}")
        
        with open("forklift_assessment_result.json", 'w', encoding='utf-8') as f:
            json.dump(assessment.model_dump(), f, ensure_ascii=False, indent=2)
    
    elif choice == "3":
        batch_analyze_all_forklifts()
        
    elif choice == "4":
        return
    
    else:
        main()

def batch_analyze_all_forklifts():
    testcup_path = Path("testCUP")
    
    if not testcup_path.exists():
        return
    
    forklift_folders = [f for f in testcup_path.iterdir() if f.is_dir()]
    forklift_folders.sort(key=lambda x: int(x.name) if x.name.isdigit() else float('inf'))
    
    all_results = []
    
    for i, folder in enumerate(forklift_folders, 1):
        forklift_id = folder.name
        
        photo_paths = get_forklift_photos(forklift_id)
        if not photo_paths:
            continue
        
        try:
            assessment = analyze_forklift(photo_paths)
            
            result = assessment.model_dump()
            result['forklift_id'] = forklift_id
            all_results.append(result)
            
            output_file = f"forklift_{forklift_id}_assessment.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            continue
    
    if all_results:
        with open("batch_assessment_results.json", 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
