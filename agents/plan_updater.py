# agents/plan_updater.py
import json
import os
from datetime import datetime
from langsmith import traceable

@traceable(name="PlanUpdater", run_type="chain", tags=["planner", "update"])
def update_plan_after_post(state):
    """
    Aggiorna il piano editoriale dopo che un post è stato approvato
    Questo agente dovrebbe essere chiamato dopo human_review quando approved
    """
    
    print("\n" + "="*50)
    print("📝 PLAN UPDATER - Marking topic as completed")
    print("="*50)
    
    memory_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "editorial_memory.json")
    )
    
    completed_topic = state.get('current_topic')
    final_post = state.get('final_post', {})
    
    if not completed_topic:
        print("⚠️ No topic to mark as completed")
        return state
    
    # Carica memoria
    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except:
        print("⚠️ Could not load memory")
        return state
    
    # Trova il piano attivo e marca il topic come finito
    updated = False
    now_iso = datetime.utcnow().isoformat() + "Z"
    
    for plan_idx, plan in enumerate(memory.get("plans", [])):
        if plan.get("finished", False):
            continue
        
        # Cerca il topic nel piano
        for topic_idx, topic_item in enumerate(plan.get("topics", [])):
            topic_name = topic_item.get("topic")
            
            if topic_name == completed_topic and not topic_item.get("finished", False):
                # Marca come finito
                topic_item["finished"] = True
                topic_item["finished_at"] = now_iso
                plan["last_topic_index"] = topic_idx
                print(f"✅ Marked topic as completed: {completed_topic}")
                updated = True
                
                # Verifica se tutti i topic sono finiti
                all_finished = all(t.get("finished", False) for t in plan.get("topics", []))
                if all_finished:
                    plan["finished"] = True
                    plan["finished_at"] = now_iso
                    print(f"🎉 Plan {plan_idx + 1} is now COMPLETE!")
                break
        
        if updated:
            break
    
    if updated:
        # Salva memoria aggiornata
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f"💾 Updated memory saved")
    else:
        print(f"⚠️ Topic '{completed_topic}' not found in active plan")
    
    print("="*50)
    
    return state