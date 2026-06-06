# test_plan_update.py
import json
import os
from datetime import datetime

def test_update():
    """Test per verificare l'aggiornamento del piano"""
    
    memory_path = "data/editorial_memory.json"
    os.makedirs("data", exist_ok=True)
    
    # Crea un piano di test
    test_plan = {
        "domain": "sport",
        "plans": [
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "finished": False,
                "last_topic_index": -1,
                "topics": [
                    {"index": 0, "topic": "Topic Test 1", "finished": False, "finished_at": None},
                    {"index": 1, "topic": "Topic Test 2", "finished": False, "finished_at": None}
                ]
            }
        ]
    }
    
    with open(memory_path, "w") as f:
        json.dump(test_plan, f, indent=2)
    
    print("Test plan created:")
    print(json.dumps(test_plan, indent=2))
    
    # Simula completamento del primo topic
    with open(memory_path, "r") as f:
        memory = json.load(f)
    
    for plan in memory.get("plans", []):
        for topic in plan.get("topics", []):
            if topic["topic"] == "Topic Test 1":
                topic["finished"] = True
                topic["finished_at"] = datetime.utcnow().isoformat() + "Z"
                plan["last_topic_index"] = 0
                print(f"\n✅ Updated: {topic['topic']} -> finished")
    
    with open(memory_path, "w") as f:
        json.dump(memory, f, indent=2)
    
    print("\nUpdated plan:")
    with open(memory_path, "r") as f:
        print(json.dumps(json.load(f), indent=2))

if __name__ == "__main__":
    test_update()