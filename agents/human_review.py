# agents/human_review.py
from typing import Dict

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
        preview = content[:800] + "..." if len(content) > 800 else content
        print(f"\n📖 CONTENT PREVIEW:\n{preview}")
        
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
        print("2. MODIFY - Edit and regenerate")
        print("3. REJECT - Discard and research again")
        print("4. SUGGEST - Provide specific changes")
        
        choice = input("\nYour choice (1/2/3/4): ").strip()
        
        if choice == "1":
            return {"action": "approve", "feedback": None}
        elif choice == "2":
            modifications = input("Enter modifications needed: ")
            return {"action": "modify", "feedback": modifications}
        elif choice == "3":
            return {"action": "reject", "feedback": None}
        elif choice == "4":
            suggestions = input("Enter specific suggestions: ")
            return {"action": "suggest", "feedback": suggestions}
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
                    topics = [state.get('current_topic', 'General')]
                    source_urls = []
                    for s in post.get('sources', []):
                        if isinstance(s, dict) and s.get('url'):
                            source_urls.append(s['url'])
                        elif isinstance(s, str):
                            source_urls.append(s)
                    
                    post_id = kg.add_post(
                        title=post.get('title', 'Untitled'),
                        content=post.get('content', ''),
                        topics=topics,
                        sources=source_urls
                    )
                    print(f"\n✅ Post saved to Knowledge Graph (ID: {post_id})")
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
            'final_post': post
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
    
    elif review_result['action'] == 'suggest':
        return {
            'review_action': 'suggestions_given',
            'suggestions': review_result['feedback'],
            'requires_regeneration': True
        }
    
    else:
        return {
            'review_action': 'rejected',
            'requires_research': True
        }