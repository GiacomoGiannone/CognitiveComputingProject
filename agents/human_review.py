# agents/human_review.py
import json
import re
import os
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langsmith import traceable
from typing import Dict
from tools.entity_extractor import extract_topics_from_title

class HumanReviewAgent:
    def __init__(self):
        self.review_state = {}
    
    @traceable(name="HumanReview-PresentForReview", run_type="chain", tags=["human_review"])
    def present_for_review(self, post: Dict, fact_check_results: Dict, state: Dict = None) -> Dict:
        """Presenta il post per review umana"""
        
        print("\n" + "="*60)
        print(" POST READY FOR REVIEW")
        print("="*60)
        print(f"\n TITLE: {post.get('title', 'No title')}")
        
        # Prendi il content e mostra solo anteprima se troppo lungo
        content = post.get('content', 'No content')
        #preview = content[:800] + "..." if len(content) > 800 else content
        print(f"\n CONTENT PREVIEW:\n{content}")
        
        sources = post.get('sources', [])
        print(f"\n SOURCES: {len(sources)} sources")
        
        if fact_check_results:
            print(f"\n FACT CHECK: {fact_check_results.get('claims_checked', 0)} claims checked")
            if fact_check_results.get('issues_found'):
                print(f" ISSUES FOUND: {fact_check_results['issues_found']}")
                print(f" SUGGESTIONS: {fact_check_results.get('suggestions', '')[:500]}")
        
        if state:
            quality_passed = state.get('quality_passed', True)
            barely_passed = state.get('barely_passed', False)
            threshold = 0.70
            print("\n\n============================================================")
            print("Now evaluate whether to publish, modify or regenerate the post\nYou can rely on the quality metric computed before by the score agent to make your decision:")
            if quality_passed:
                print(f"\n ✅ Good quality (score >= {threshold}), recommended to publish.")
            elif not quality_passed:
                print(f"\n ❌ Insufficient quality (score < {threshold}), recommended to regenerate!")
            elif barely_passed:
                print(f"\n ⚠️ Marginal quality (score between 0.60 and {threshold}), consider human revision...")

        print("\n" + "="*60)
        print("Options:")
        print("1. APPROVE - Publish as is")
        print("2. MODIFY - Provide feedback or corrections to regenerate")
        print("3. REJECT - Discard and research again")
        
        choice = input("\nYour choice (1/2/3): ").strip()
        
        if choice == "1":
            proceed_choice = input("Vuoi procedere subito alla ricerca del prossimo topic del piano editoriale? (s/n): ").strip().lower()
            proceed = proceed_choice in ("s", "si")
            return {"action": "approve", "feedback": None, "proceed": proceed}
        elif choice == "2":
            modifications = input("Enter modifications needed: ")
            return {"action": "modify", "feedback": modifications}
        elif choice == "3":
            return {"action": "reject", "feedback": None}
        else:
            return {"action": "reject", "feedback": "Invalid choice"}


@traceable(name="PlanUpdater", run_type="chain", tags=["planner", "update"])
def _update_plan_after_post(state):
    """
    Aggiorna il piano editoriale dopo che un post è stato approvato.
    Marca il topic corrente come completato nella memoria editoriale.
    """
    
    print("\n" + "="*50)
    print(" PLAN UPDATER - Marking topic as completed")
    print("="*50)
    
    memory_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data", "editorial_memory.json")
    )
    
    completed_topic = state.get('current_topic')
    
    if not completed_topic:
        print(" No topic to mark as completed")
        return state
    
    # Carica memoria
    try:
        with open(memory_path, "r", encoding="utf-8") as f:
            memory = json.load(f)
    except:
        print(" No topic to mark as completed")
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
                print(f" Marked topic as completed: {completed_topic}")
                updated = True
                
                # Verifica se tutti i topic sono finiti
                all_finished = all(t.get("finished", False) for t in plan.get("topics", []))
                if all_finished:
                    plan["finished"] = True
                    plan["finished_at"] = now_iso
                    print(f" Plan {plan_idx + 1} is now COMPLETE!")
                break
        
        if updated:
            break
    
    if updated:
        # Salva memoria aggiornata
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
        print(f" Updated memory saved")
    else:
        print(f" No topic to mark as completed")
    
    print("="*50)
    
    return state


@traceable(name="HumanReviewAgent", run_type="chain", tags=["agent", "human_review"])
def human_review_agent(state):
    """Human review step nel workflow"""
    reviewer = HumanReviewAgent()
    
    post = state.get('draft_post', {})
    fact_check = state.get('fact_check_results', {})
    
    review_result = reviewer.present_for_review(post, fact_check, state)
    
    if review_result['action'] == 'approve':
        # Aggiorna KG
        kg = state.get('kg_manager')
        post_id = None
        
        if kg:
            try:
                if hasattr(kg, 'add_post'):
                    post_title = post.get('title', 'Untitled')
                    extracted_topics = state.get('extracted_graph_topics', [])  
                    
                    print("\n" + "="*60)
                    print(" Saving to Knowledge Graph")
                    print("="*60)
                    print(f" Post title: {post_title}")
                    print(f"   - Using pre-extracted topics: {extracted_topics}")  
                    print("="*60)
                    
                    # Raccogliere source URLs
                    source_urls = []
                    for s in post.get('sources', []):
                        if isinstance(s, dict) and s.get('url'):
                            source_urls.append(s['url'])
                        elif isinstance(s, str):
                            source_urls.append(s)
                    
                    # Recupera i claims estratti dal fact checker
                    extracted_claims = fact_check.get('extracted_claims', [])
                    
                    # Salva in Neo4j con il titolo lungo e i topic generici estratti (dallo state)
                    post_id = kg.add_post(
                        title=post_title,  # Titolo specifico e lungo
                        content=post.get('content', ''),
                        topics=extracted_topics,  # Topic generici dallo state, non ri-estratti
                        sources=source_urls,
                        claims=extracted_claims
                    )
                    print(f"\n Post saved to Knowledge Graph")
                    if extracted_claims:
                        print(f"    {len(extracted_claims)} claims saved as Claim nodes")
                    
                    # CREAZIONE RELAZIONI TRA TOPIC
                    print(f"\n Creating Topic Relations...")
                    try:
                        # Recupera i topic storici già nel grafo
                        all_existing_topics = kg.get_covered_topics()
                        print(f"    Looking for related topics")
                        
                        if all_existing_topics and extracted_topics:
                            # Chiedi all'LLM di valutare relazioni semantiche
                            new_topics_str = ", ".join(extracted_topics)
                            #analizziamo tutti i topic esistenti, non gli ultimi X
                            existing_str = ", ".join(all_existing_topics)  # Tutti i topic esistenti
                            
                            relation_prompt = f"""
                            Analizza le relazioni semantiche tra questi topic:
                            
                            NEW topics appena salvati: [{new_topics_str}]
                            EXISTING topics in KG: [{existing_str}]
                            
                            Identifica le relazioni logiche/semantiche (padre-figlio, affinità tematica, correlazione).
                            Ritorna SOLO un array JSON di coppie ["topic1", "topic2"] senza ulteriore testo.
                            
                            Esempio:
                            [["Ansia", "Psicologia sportiva"], ["Vela", "Vento"]]
                            
                            Se non trovi relazioni, ritorna: []
                            """
                            
                            llm = ChatOllama(model="llama3.1", temperature=0)
                            response = llm.invoke([HumanMessage(content=relation_prompt)])
                            
                            # Parse JSON delle relazioni con fallback
                            relations = []
                            try:
                                content = response.content
                                # Estrai blocco JSON
                                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                                if json_match:
                                    relations = json.loads(json_match.group())
                            except (json.JSONDecodeError, AttributeError) as e:
                                print(f"    Could not parse relations JSON: {e}")
                            
                            # Crea le relazioni tra Topic nel grafo
                            if relations:
                                print(f"    Found {len(relations)} relations to create:")
                                for topic1, topic2 in relations:
                                    try:
                                        kg.add_topic_relation(topic1, topic2, "RELATED_TO")
                                        print(f"       {topic1} <-> {topic2}")
                                    except Exception as rel_err:
                                        print(f"       Could not create relation: {rel_err}")
                            else:
                                print(f"    No semantic relations found")
                            
                            # CONNESSIONE POST → TOPIC ESISTENTI
                            # Identifica i topic esistenti coinvolti nelle relazioni trovate
                            # e collega il Post direttamente a quei topic
                            existing_set = set(all_existing_topics) #topic già presenti nel grafo
                            new_set = set(extracted_topics) #topic che bisogna salvare nel grafo
                            connected_existing = set() 
                            
                            for pair in relations:
                                if len(pair) == 2:
                                    for t in pair:
                                        # Collega solo se è un topic esistente e NON è già
                                        # tra i topic del post (quelli hanno già COVERS)
                                        if t in existing_set and t not in new_set:
                                            connected_existing.add(t)
                            
                            if connected_existing and post_id:
                                print(f"\n    Connecting Post to {len(connected_existing)} existing topics:")
                                for existing_topic in sorted(connected_existing):
                                    try:
                                        kg.connect_post_to_topic(post_id, existing_topic, "COVERS")
                                        print(f"       Post → COVERS → {existing_topic}")
                                    except Exception as conn_err:
                                        print(f"       Could not connect post to '{existing_topic}': {conn_err}")
                            else:
                                print(f"\n    No existing topics to connect the post to")
                        else:
                            print(f"    Not enough topics to establish relations")
                    except Exception as rel_err:
                        print(f"   Relation creation error: {rel_err}")
            except Exception as e:
                print(f"\n   Could not save to KG: {e}")
        
        # Aggiorna il piano editoriale
        try:
            state = _update_plan_after_post(state)
        except Exception as e:
            print(f" Could not update plan: {e}")
        
        return {
            'review_action': 'approved',
            'final_post': post,
            'proceed_to_next_topic': review_result.get('proceed', False)
        }
    
    elif review_result['action'] == 'modify':
        # Ritorna al writer con modifiche
        return {
            'review_action': 'modify_requested',
            'modification_feedback': review_result['feedback']
        }
    
    elif review_result['action'] == 'reject':
        # Ritorna al research
        return {
            'review_action': 'rejected',
            'requires_research': True,
            'iteration': state.get('iteration', 0) + 1
        }
    
    else:
        return {
            'review_action': 'rejected',
            'requires_research': True
        }