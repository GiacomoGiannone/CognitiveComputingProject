# utils/show_plan_status.py
import json
import os

def show_plan_status():
    """Utility per visualizzare lo stato corrente del piano editoriale"""
    
    memory_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "editorial_memory.json")
    )
    
    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except:
        print("No editorial plan found")
        return
    
    print("\n" + "="*60)
    print("📅 EDITORIAL PLAN STATUS")
    print("="*60)
    
    for plan_idx, plan in enumerate(memory.get("plans", [])):
        print(f"\n📋 PLAN {plan_idx + 1} - Created: {plan.get('created_at', 'Unknown')[:19]}")
        print(f"   Status: {'✅ COMPLETE' if plan.get('finished') else '🔄 IN PROGRESS'}")
        print(f"   Last topic index: {plan.get('last_topic_index', -1)}")
        print("\n   Topics:")
        
        for topic_item in plan.get("topics", []):
            status = "✅" if topic_item.get("finished") else "⏳"
            finished_at = f" - finished: {topic_item.get('finished_at', '')[:19]}" if topic_item.get("finished_at") else ""
            print(f"      {status} {topic_item.get('topic')}{finished_at}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    show_plan_status()