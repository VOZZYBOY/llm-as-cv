import os
import json
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
from openai import OpenAI
from gemini import ForkliftAssessment, analyze_forklift

class DeepSeekJudge:
    def __init__(self):
        self.client = OpenAI(
            api_key="api-kry",
            base_url="https://api.deepseek.com"
        )
        
    def load_csv_data(self, csv_path: str = "testCUP.csv") -> pd.DataFrame:
        df = pd.read_csv(csv_path, encoding='utf-8')
        df = df.iloc[:, :-1]
        return df
    
    def get_ground_truth_for_forklift(self, forklift_id: str, df: pd.DataFrame) -> Optional[Dict]:
        for idx, row in df.iterrows():
            if str(row.iloc[0]) == forklift_id:
                return row.to_dict()
        return None
    
    def judge_assessment(self, forklift_id: str, model_result: ForkliftAssessment, ground_truth: Dict) -> Dict:
        system_prompt = """You are an expert judge evaluating forklift assessment models. Compare model predictions with ground truth data and provide detailed evaluation in JSON format.

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
}"""

        user_prompt = f"""Evaluate this forklift assessment:

FORKLIFT ID: {forklift_id}

MODEL ASSESSMENT RESULT:
- Cleanliness: {model_result.cleanliness} (0=problem, 1=good, -1=absent)
- Paint condition: {model_result.paint_condition} (0=problem, 1=good, -1=absent)
- Mirrors: {model_result.mirrors} (0=problem, 1=good, -1=absent)
- Lights: {model_result.lights} (0=problem, 1=good, -1=absent)
- Seat: {model_result.seat} (0=problem, 1=good, -1=absent)
- Roof: {model_result.roof} (0=problem, 1=good, -1=absent)
- Glass: {model_result.glass} (0=problem, 1=good, -1=absent)
- Tires: {model_result.tires} (0=problem, 1=good, -1=absent)
- Other breakdowns: {model_result.other_breakdowns} (0=problem, 1=good, -1=absent)
- Photo rules: {model_result.photo_rules} (0=problem, 1=good, -1=absent)

GROUND TRUTH FROM CSV:
{json.dumps({k: v for k, v in ground_truth.items() if not pd.isna(v)}, ensure_ascii=False, indent=2)}

CSV VALUE MAPPING:
- "1" or 1 = good condition (1)
- "0" or 0 = problem detected (0)  
- "нет" or "absent" = component absent (-1)

Evaluate accuracy of each parameter, identify critical errors (safety-related), minor issues, and provide overall assessment score. Output as JSON format matching the example structure."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        judgment_text = response.choices[0].message.content
        return json.loads(judgment_text)
    
    def analyze_forklift_with_judge(self, forklift_id: str) -> Dict:
        folder_path = Path(f"testCUP/{forklift_id}")
        photo_paths = list(folder_path.glob("*.jpg"))[:4]
        photo_paths = [str(path) for path in photo_paths]
        
        model_result = analyze_forklift(photo_paths)
        
        df = self.load_csv_data()
        ground_truth = self.get_ground_truth_for_forklift(forklift_id, df)
        
        if ground_truth is None:
            return {
                "error": f"No ground truth found for forklift {forklift_id}",
                "model_result": model_result.model_dump()
            }
        
        judgment = self.judge_assessment(forklift_id, model_result, ground_truth)
        
        return {
            "forklift_id": forklift_id,
            "model_result": model_result.model_dump(),
            "ground_truth": {k: v for k, v in ground_truth.items() if not pd.isna(v)},
            "judge_evaluation": judgment,
            "photos_analyzed": len(photo_paths)
        }
    
    def batch_judge_analysis(self, forklift_ids: List[str]) -> Dict:
        results = []
        total_score = 0
        critical_errors_count = 0
        
        for forklift_id in forklift_ids:
            print(f"Analyzing forklift {forklift_id} with DeepSeek judge...")
            result = self.analyze_forklift_with_judge(forklift_id)
            
            if "error" not in result:
                results.append(result)
                total_score += result["judge_evaluation"]["overall_score"]
                critical_errors_count += len(result["judge_evaluation"]["critical_errors"])
                
                print(f"  Score: {result['judge_evaluation']['overall_score']:.2f}")
                if result["judge_evaluation"]["critical_errors"]:
                    print(f"  Critical errors: {len(result['judge_evaluation']['critical_errors'])}")
            else:
                print(f"  Error: {result['error']}")
        
        avg_score = total_score / len(results) if results else 0
        
        summary = {
            "total_analyzed": len(results),
            "average_score": avg_score,
            "total_critical_errors": critical_errors_count,
            "results": results
        }
        
        return summary

def main():
    judge = DeepSeekJudge()
    print("1. Judge single forklift")
    print("2. Batch judge multiple forklifts")
    print("3. Judge all available forklifts")
    
    choice = input("Choose mode (1-3): ").strip()
    
    if choice == "1":
        forklift_id = input("Enter forklift ID: ").strip()
        result = judge.analyze_forklift_with_judge(forklift_id)
        
        print(f"\nJudgment for forklift {forklift_id}:")
        print("=" * 40)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        output_file = f"judge_result_{forklift_id}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nResult saved to: {output_file}")
        
    elif choice == "2":
        ids_input = input("Enter forklift IDs separated by commas: ").strip()
        forklift_ids = [id.strip() for id in ids_input.split(",")]
        
        summary = judge.batch_judge_analysis(forklift_ids)
        
        print(f"\nBatch Analysis Summary:")
        print("=" * 40)
        print(f"Total analyzed: {summary['total_analyzed']}")
        print(f"Average score: {summary['average_score']:.2f}")
        print(f"Critical errors: {summary['total_critical_errors']}")
        
        output_file = "batch_judge_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to: {output_file}")
        
    elif choice == "3":
        testcup_path = Path("testCUP")
        forklift_folders = [f.name for f in testcup_path.iterdir() if f.is_dir()]
        forklift_folders.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        print(f"Found {len(forklift_folders)} forklifts")
        confirm = input("Analyze all? (y/n): ").strip().lower()
        
        if confirm == 'y':
            summary = judge.batch_judge_analysis(forklift_folders)
            
            print(f"\nComplete Analysis Summary:")
            print("=" * 40)
            print(f"Total analyzed: {summary['total_analyzed']}")
            print(f"Average score: {summary['average_score']:.2f}")
            print(f"Critical errors: {summary['total_critical_errors']}")
            
            output_file = "complete_judge_analysis.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
