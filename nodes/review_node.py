def review_node(state):
    print("\n" + "="*50)
    print("=== BOZZA ARTICOLO GENERATA DAL WRITER ===")
    print("="*50)
    print(state.get("created_content", "Nessun contenuto generato."))
    print("="*50 + "\n")

    print("[review] start")

    feedback = input("Approve post? (yes/no): ").strip().lower()
    feedback_detail = ""
    if feedback != "yes":
        feedback_detail = input("Inserisci feedback per migliorare il post: ").strip()

    state["content_feedback"] = feedback
    state["content_feedback_detail"] = feedback_detail

    if state["content_feedback"] != "yes":
        detail = state.get("content_feedback_detail", "").strip()
        if detail:
            state["revision_query"] = detail

    print(f"[review] feedback={state['content_feedback']}")

    return state