import google.genai as genai

from kg.KG import KG

from agents import planner_agent

import ollama

def main():
	#Inizializza il Knowledge Graph
	kg = KG()

	#Inizializza lo stato degli agenti
	state = {
		"user_input": "Voglio saperne di più sull'intelligenza artificiale",
		"recent_topics": [],
		"chosen_topic": None,
		"verified_info": None,
		"reasoning_trace": None,
		"created_content": None,
		"content_feedback": None
	}

	#Aggiungiamo un nodo di esempio al KG e vediamo se il modello capisce che non deve 
	#scegliere i topic gia' coperti di recente
	kg.add_node("topic_1", "Topic", {"name": "Intelligenza Artificiale", "description": "Il campo dell'informatica che si occupa di creare sistemi in grado di svolgere compiti che richiederebbero intelligenza umana."})
	kg.add_node("post_1", "Post", {"title": "Introduzione all'Intelligenza Artificiale", "date": "2026-05-21"})
	kg.add_relationship("post_1", "COVERS", "topic_1")

	#Inizializza il modello (sostituisci con il tuo modello preferito)
	model = ollama.Client()
	model_name = "llama3.1"

	#Inizializza il planner agent
	planner = planner_agent.Planner_agent(state, model, model_name=model_name)

	#Esempio di utilizzo del planner agent per scegliere un topic
	chosen_topic = planner.choose_topic()
	print(f"Planner agent dice: {chosen_topic}")

if __name__ == "__main__":
	main()

