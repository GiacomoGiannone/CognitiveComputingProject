# fact_check_agent.py
import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langsmith import traceable
from tools.tavily_search import web_search
from typing import List, Dict

class FactCheckAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm_model = llm_model

    def _chat(self, prompt: str) -> str:
        llm = ChatOllama(model=self.llm_model, temperature=0)
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    @traceable(name="FactCheck-ExtractClaims", run_type="chain", tags=["fact_check", "claims"])
    def extract_claims(self, post_content: str) -> List[str]:
        """Estrae claims verificabili dal post - SOLO fatti specifici"""
        prompt = f"""
        Extract ONLY verifiable factual claims from this blog post.
        DO NOT include:
        - Introductory text or headers
        - Generic statements ("The blog covers...", "This post discusses...")
        - Markdown formatting or bullet points
        - Opinions or advice
        
        Return a JSON list of strings containing only the specific factual claims.
        
        Example format:
        [
            "Deep breathing reduces anxiety by 40%",
            "Aerobic training improves heart health",
            "Sleep deprivation impacts cognitive performance"
        ]
        
        Post:
        {post_content[:3000]}
        
        Return ONLY the JSON list, nothing else.
        """
        
        response = self._chat(prompt)
        claims = []
        try:
            # Estrai il blocco JSON dalla risposta (tra parentesi quadre)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                claims = json.loads(json_str)
            else:
                # Fallback on splitting lines if no JSON block is found
                print(f"    No JSON list found in response, falling back to line splitting")
                claims = [c.strip() for c in response.split('\n') if c.strip()]
        except Exception as e:
            print(f"    Claim extraction parse error: {e}, falling back to line splitting")
            claims = [c.strip() for c in response.split('\n') if c.strip()]
            
        # Pulisci claims (rimuovi righe vuote o introduttive)
        cleaned_claims = []
        for c in claims:
            if not isinstance(c, str):
                continue
            c_clean = c.strip()
            if not c_clean:
                continue
            # Salta frasi introduttive comuni
            lower_c = c_clean.lower()
            if (lower_c.startswith("here are") or 
                lower_c.startswith("here is") or 
                lower_c.startswith("factual claims") or 
                lower_c.startswith("verifiable claims") or
                lower_c.startswith("based on") or
                lower_c.endswith("extracted from the blog post:") or
                lower_c.endswith("extracted from the post:")):
                continue
            cleaned_claims.append(c_clean)
            
        return cleaned_claims[:5]  # Limita a 5 claims
    
    @traceable(name="FactCheck-VerifyClaim", run_type="chain", tags=["fact_check", "verification"])
    def verify_claim(self, claim: str) -> Dict:
        """Verifica un claim specifico con parsing JSON rigoroso"""
        search_query = f"verify fact: {claim}"
        search_results = web_search.invoke({"query": search_query, "max_results": 3})
        
        verification_prompt = f"""
        Claim to verify: "{claim}"
        
        Search results:
        {search_results}
        
        Analyze and return ONLY a valid JSON object (no markdown, no other text) with exactly these keys:
        - "supported": boolean (true if claim is supported by search results, false otherwise)
        - "confidence": integer from 0 to 100 (the level of certainty of the answer)
        - "correction": string with suggested correction if needed, or "N/A" if supported
        
        Example format:
        {{"supported": true, "confidence": 85, "correction": "N/A"}}
        {{"supported": false, "confidence": 45, "correction": "The correct fact is..."}}
        
        Return ONLY the JSON object, nothing else.
        """
        
        response = self._chat(verification_prompt)
        
        # Parse JSON rigorosamente con try/except
        verification_data = {
            "supported": None,
            "confidence": 0,
            "correction": "Parse error"
        }
        
        try:
            # Estrai il blocco JSON dalla risposta (tra parentesi graffe)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                verification_data = json.loads(json_str)
            else:
                print(f"    No JSON block found in response")
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e}")
        except Exception as e:
            print(f"    Verification parse error: {e}")
        
        return {
            'claim': claim,
            'verification': verification_data,
            'sources': search_results.get('results', [])
        }
    
    @traceable(name="FactCheck-FullPost", run_type="chain", tags=["fact_check"])
    def fact_check_post(self, post_content: str) -> Dict:
        """Fact-check completo del post"""
        claims = self.extract_claims(post_content)
        print(f"\n Extracted {len(claims)} claims to verify")
        
        verified_claims = []
        issues_found = []
        
        for idx, claim in enumerate(claims, 1):
            print(f"\n   [{idx}/{len(claims)}] Verifying: {claim}...")
            result = self.verify_claim(claim)
            verified_claims.append(result)
            
            # Valuta il booleano dal JSON parsato
            is_supported = result['verification'].get('supported')
            confidence = result['verification'].get('confidence', 0)
            
            if is_supported is False:
                print(f"        ❌ NOT SUPPORTED (confidence: {confidence}%)")
                issues_found.append(claim)
            elif is_supported is True:
                print(f"        ✅ SUPPORTED (confidence: {confidence}%)")
            else:
                print(f"       ❔  UNCLEAR (parse error)")
        
        # Genera suggerimenti
        suggestions = ""
        if issues_found:
            print(f"\n Found {len(issues_found)} unsupported claims, generating suggestions...")
            suggestions_prompt = f"""
            Based on fact-checking results:
            
            Issues found: {issues_found}
            
            Generate specific suggestions to fix these claims in the post.
            Provide corrected versions where possible.
            """
            
            suggestions = self._chat(suggestions_prompt)
            print(f" Suggestions generated: {suggestions}")
        else:
            print(f"\n All claims verified successfully!")
        
        return {
            'claims_checked': len(claims),
            'issues_found': issues_found,
            'suggestions': suggestions,
            'detailed_results': verified_claims,
            'extracted_claims': claims
        }


@traceable(name="FactCheckAgent", run_type="chain", tags=["agent", "fact_check"])
def fact_check_agent(state):
    """Fact check agent per workflow"""
    print("\n" + "="*60)
    print("🕵️ FACT CHECK AGENT STARTING")
    print("="*60)
    
    checker = FactCheckAgent()
    
    draft = state.get('draft_post', {})
    content = draft.get('content', '') if isinstance(draft, dict) else draft
    
    if not content:
        print(" No content to fact-check")
        return {
            'fact_check_passed': True,
            'fact_check_results': {'claims_checked': 0, 'issues_found': [], 'suggestions': '', 'detailed_results': []},
            'requires_revision': False
        }
    
    fact_check_result = checker.fact_check_post(content)
    
    print(f"\n" + "="*60)
    print(f" FACT CHECK SUMMARY")
    print(f"   - Claims checked: {fact_check_result['claims_checked']}")
    print(f"   - Issues found: {len(fact_check_result['issues_found'])}")
    print(f"   - Fact check passed: {len(fact_check_result['issues_found']) == 0}")
    print("="*60)
    
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