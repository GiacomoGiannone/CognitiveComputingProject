# fact_check_agent.py
import ollama
from tools.tavily_search import web_search
from typing import List, Dict

class FactCheckAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm_model = llm_model

    def _chat(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]
    
    def extract_claims(self, post_content: str) -> List[str]:
        """Estrae claims verificabili dal post"""
        prompt = f"""
        Extract verifiable claims from this blog post.
        Return each claim on a new line.
        
        Post:
        {post_content[:3000]}
        """
        
        response = self._chat(prompt)
        claims = [c.strip() for c in response.split('\n') if c.strip()]
        
        return claims[:10]  # Limita a 10 claims
    
    def verify_claim(self, claim: str) -> Dict:
        """Verifica un claim specifico"""
        search_query = f"verify fact: {claim}"
        search_results = web_search.invoke({"query": search_query, "max_results": 3})
        
        verification_prompt = f"""
        Claim to verify: "{claim}"
        
        Search results:
        {search_results}
        
        Analyze:
        1. Is this claim supported by the search results?
        2. Rate confidence (0-100%)
        3. Suggest corrections if needed
        4. Provide supporting sources
        
        Return JSON with: supported, confidence, correction, sources
        """
        
        response = self._chat(verification_prompt)
        
        return {
            'claim': claim,
            'verification': response,
            'sources': search_results.get('results', [])
        }
    
    def fact_check_post(self, post_content: str) -> Dict:
        """Fact-check completo del post"""
        claims = self.extract_claims(post_content)
        
        verified_claims = []
        issues_found = []
        
        for claim in claims:
            result = self.verify_claim(claim)
            verified_claims.append(result)
            
            # Check if verification found issues
            if "not supported" in result['verification'].lower():
                issues_found.append(claim)
        
        # Genera suggerimenti
        suggestions_prompt = f"""
        Based on fact-checking results:
        
        Issues found: {issues_found}
        
        Generate specific suggestions to fix these claims in the post.
        Provide corrected versions where possible.
        """
        
        suggestions = self._chat(suggestions_prompt)
        
        return {
            'claims_checked': len(claims),
            'issues_found': issues_found,
            'suggestions': suggestions,
            'detailed_results': verified_claims
        }


def fact_check_agent(state):
    """Fact check agent per workflow"""
    checker = FactCheckAgent()
    
    draft = state.get('draft_post', {})
    content = draft.get('content', '') if isinstance(draft, dict) else draft
    
    fact_check_result = checker.fact_check_post(content)
    
    # Se ci sono problemi, suggerisci modifiche
    if fact_check_result['issues_found']:
        return {
            'fact_check_passed': False,
            'fact_check_results': fact_check_result,
            'requires_revision': True
        }
    else:
        return {
            'fact_check_passed': True,
            'fact_check_results': fact_check_result,
            'requires_revision': False
        }