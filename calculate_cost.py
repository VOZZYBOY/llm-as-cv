import json
from pathlib import Path

def run_batch_analysis():
    testcup_path = Path("testCUP")
    if not testcup_path.exists():
        return
    
    forklift_folders = [f for f in testcup_path.iterdir() if f.is_dir()]
    forklift_folders.sort(key=lambda x: int(x.name) if x.name.isdigit() else float('inf'))
    
    with open("usage_log.jsonl", "w", encoding="utf-8") as f:
        f.write("")
    
    from gemini import get_forklift_photos, analyze_forklift
    
    for i, folder in enumerate(forklift_folders, 1):
        forklift_id = folder.name
        
        photo_paths = get_forklift_photos(forklift_id)
        if not photo_paths:
            continue
            
        try:
            assessment = analyze_forklift(photo_paths)
        except Exception as e:
            continue

def calculate_cost():
    flash_lite_input_price = 0.10 / 1000000
    flash_lite_output_price = 0.40 / 1000000
    
    pro_input_price_under_200k = 1.25 / 1000000
    pro_output_price_under_200k = 10.00 / 1000000
    pro_input_price_over_200k = 2.50 / 1000000
    pro_output_price_over_200k = 15.00 / 1000000
    
    total_cost = 0
    forklift_costs = {}
    
    with open("usage_log.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            prompt_tokens = data["prompt_tokens"]
            completion_tokens = data["completion_tokens"]
            model = data["model"]
            forklift_id = data["forklift_id"]
            
            if "flash-lite" in model:
                cost = (prompt_tokens * flash_lite_input_price) + (completion_tokens * flash_lite_output_price)
            elif "pro" in model:
                total_tokens = prompt_tokens + completion_tokens
                if total_tokens <= 200000:
                    cost = (prompt_tokens * pro_input_price_under_200k) + (completion_tokens * pro_output_price_under_200k)
                else:
                    cost = (prompt_tokens * pro_input_price_over_200k) + (completion_tokens * pro_output_price_over_200k)
            
            total_cost += cost
            
            if forklift_id not in forklift_costs:
                forklift_costs[forklift_id] = 0
            forklift_costs[forklift_id] += cost
    
    total_forklifts = len(forklift_costs)
    avg_cost_per_forklift = total_cost / total_forklifts if total_forklifts > 0 else 0
    
    return {
        "total_forklifts": total_forklifts,
        "total_cost": total_cost,
        "avg_cost_per_forklift": avg_cost_per_forklift,
        "forklift_costs": forklift_costs
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        run_batch_analysis()
    else:
        result = calculate_cost()
        if result:
            print(f"Total: {result['total_forklifts']} forklifts, ${result['total_cost']:.6f}, avg: ${result['avg_cost_per_forklift']:.6f}")
