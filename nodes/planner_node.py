from agents.planner_agent import Planner_agent

def planner_node(state):
    print("[planner] start")
    print(f"[planner] state={state}")
    model = state.get("model", "llama3.1")
    model_name = state.get("model_name", "llama3.1")
    kg = state.get("kg")

    planner = Planner_agent(
        state=state,
        model=model,
        kg=kg,
        model_name=model_name
    )

    chosen_topic = planner.choose_topic()

    print(f"[planner] chosen_topic={chosen_topic}")
    if state.get("planner_mode") == "suggest":
        print(f"[planner] suggested_topic={state.get('suggested_topic')}")

        approve = input("Approve suggested topic? (y/n): ").strip().lower()
        if approve != "y":
            user_prompt = input("Inserisci il prompt per il modello: ")
            state["user_input"] = user_prompt
            state["planner_mode"] = "choose"
            chosen_topic = planner.choose_topic()
            print(f"[planner] chosen_topic={chosen_topic}")

    state["chosen_topic"] = chosen_topic

    print(f"[planner] state_end={state}")

    return state