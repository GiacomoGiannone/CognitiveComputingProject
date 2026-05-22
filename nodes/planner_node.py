from agents.planner_agent import Planner_agent

def planner_node(state):
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

    state["chosen_topic"] = chosen_topic

    return state