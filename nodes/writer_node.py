from agents.writer_agent import WriterAgent

def writer_node(state):
    print("[writer] start")
    print(f"[writer] state={state}")
    
    agent = WriterAgent(
        state=state, 
        model=state.get("model", "llama3.1"), 
        model_name=state.get("model_name", "llama3.1")
    )
    
    draft = agent.write_post()
    state["created_content"] = draft
    
    print("[writer] draft_ready=true")

    print(f"[writer] state_end={state}")

    return state