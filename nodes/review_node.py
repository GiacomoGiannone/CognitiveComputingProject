def review_node(state):
    print("\n" + "="*50)
    print("=== BOZZA ARTICOLO GENERATA DAL WRITER ===")
    print("="*50)
    print(state.get("created_content", "Nessun contenuto generato."))
    print("="*50 + "\n")

    print("[review] start")

    feedback = input(
        "Approve post? (yes/no): "
    )

    state["content_feedback"] = feedback

    print(f"[review] feedback={feedback}")

    return state