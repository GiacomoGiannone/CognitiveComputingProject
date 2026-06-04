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

    manual_facts = ""
    add_facts = input("Vuoi aggiungere o correggere informazioni manualmente? (y/n): ").strip().lower()
    if add_facts == "y":
        manual_facts = input("Inserisci le informazioni corrette o integrate: ").strip()

    state["content_feedback"] = feedback
    state["content_feedback_detail"] = feedback_detail
    state["manual_facts"] = manual_facts

    print(f"[review] feedback={state['content_feedback']}")

    return state