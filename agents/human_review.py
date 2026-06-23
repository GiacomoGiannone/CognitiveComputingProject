# agents/human_review.py
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from typing import Dict
from tools.entity_extractor import extract_topics_from_title

class HumanReviewAgent:
    def __init__(self):
        self.review_state = {}
    
    def present_for_review(self, post: Dict, fact_check_results: Dict, state: Dict = None) -> Dict:
        """Presenta il post per review umana"""
        
        print("\n" + "="*60)
        print("📝 POST READY FOR REVIEW")
        print("="*60)
        print(f"\n📌 TITLE: {post.get('title', 'No title')}")
        
        # Mostra solo anteprima se troppo lungo
        content = post.get('content', 'No content')
        #preview = content[:800] + "..." if len(content) > 800 else content
        print(f"\n📖 CONTENT PREVIEW:\n{content}")
        
        sources = post.get('sources', [])
        print(f"\n🔗 SOURCES: {len(sources)} sources")
        
        if fact_check_results:
            print(f"\n✅ FACT CHECK: {fact_check_results.get('claims_checked', 0)} claims checked")
            if fact_check_results.get('issues_found'):
                print(f"⚠️ ISSUES FOUND: {fact_check_results['issues_found']}")
                print(f"💡 SUGGESTIONS: {fact_check_results.get('suggestions', '')[:500]}")
        
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
    
    def apply_feedback(self, post: Dict, feedback: str, writer_agent) -> Dict:
        """Applica il feedback e rigenera il post"""
        
        modified_prompt = f"""
        Original post:
        {post.get('content', '')}
        
        User feedback for modification:
        {feedback}
        
        Please regenerate the post addressing this feedback while maintaining quality and citations.
        """
        
        # Usa il writer_agent per rigenerare
        new_post = writer_agent.write_post(
            topic=post.get('topic'),
            research_results={'research_summary': modified_prompt},
            max_length=len(post.get('content', '').split())
        )
        
        return new_post


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
                    print("🔍 Saving to Knowledge Graph")
                    print("="*60)
                    print(f"📌 Post title: {post_title}")
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
                    
                    # Salva in Neo4j con il titolo lungo e i topic generici estratti (dal state)
                    post_id = kg.add_post(
                        title=post_title,  # Titolo specifico e lungo
                        content=post.get('content', ''),
                        topics=extracted_topics,  # Topic generici dal state, non ri-estratti
                        sources=source_urls,
                        claims=extracted_claims
                    )
                    print(f"\n✅ Post saved to Knowledge Graph")
                    if extracted_claims:
                        print(f"   📋 {len(extracted_claims)} claims saved as Claim nodes")
                    
                    # CREAZIONE RELAZIONI TRA TOPIC
                    print(f"\n🔗 Creating Topic Relations...")
                    try:
                        # Recupera i topic storici già nel grafo
                        all_existing_topics = kg.get_covered_topics()
                        print(f"   📊 Looking for related topics")
                        
                        if all_existing_topics and extracted_topics:
                            # Chiedi all'LLM di valutare relazioni semantiche
                            new_topics_str = ", ".join(extracted_topics)
                            existing_str = ", ".join(all_existing_topics[-10:])  # Ultimi 10 per ridurre token
                            
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
                                print(f"   ⚠️ Could not parse relations JSON: {e}")
                            
                            # Crea le relazioni nel grafo
                            if relations:
                                print(f"   ✅ Found {len(relations)} relations to create:")
                                for topic1, topic2 in relations:
                                    try:
                                        kg.add_topic_relation(topic1, topic2, "RELATED_TO")
                                        print(f"      🔗 {topic1} <-> {topic2}")
                                    except Exception as rel_err:
                                        print(f"      ⚠️ Could not create relation: {rel_err}")
                            else:
                                print(f"   ℹ️ No semantic relations found")
                        else:
                            print(f"   ℹ️ Not enough topics to establish relations")
                    except Exception as rel_err:
                        print(f"   ⚠️ Relation creation error: {rel_err}")
            except Exception as e:
                print(f"\n⚠️ Could not save to KG: {e}")
        
        # Aggiorna il piano editoriale
        try:
            from agents.plan_updater import update_plan_after_post
            state = update_plan_after_post(state)
        except Exception as e:
            print(f"⚠️ Could not update plan: {e}")
        
        return {
            'review_action': 'approved',
            'post_id': post_id,
            'final_post': post,
            'proceed_to_next_topic': review_result.get('proceed', False)
        }
    
    elif review_result['action'] == 'modify':
        # Ritorna al writer con modifiche
        return {
            'review_action': 'modify_requested',
            'modification_feedback': review_result['feedback'],
            'requires_regeneration': True
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