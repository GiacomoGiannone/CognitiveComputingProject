import google.genai as genai

from kg.KG import KG

from agents import planner_agent

import ollama

from graph.graph import graph

def main():
	#Inizializza il Knowledge Graph
	kg = KG()
	kg.reset() #Pulisce il grafo all'inizio di ogni esecuzione, per testare da zero
	#Aggiungiamo un nodo di esempio al KG e vediamo se il modello capisce che non deve 
	#scegliere i topic gia' coperti di recente
	kg.add_node("topic_2", "Topic", {"name": "Inter", "description": "La squadra piu' bella del mondo"})
	kg.add_node("post_2", "Post", {"title": "Inter ti amo", "date": "2026-05-21"})
	kg.add_relationship("post_2", "COVERS", "topic_2")

	#Inizializza il modello (sostituisci con il tuo modello preferito)
	model = ollama.Client()
	model_name = "llama3.1"

	initial_state = {
		"user_input": "Voglio scrivere articoli su delle squadre della Serie A, ma non so da dove iniziare. Cosa mi consigli?",
		"recent_topics": [],
		"chosen_topic": None,
		"verified_info": None,
		"reasoning_trace": [],
		"created_content": None,
		"content_feedback": None,
		"kg": kg,
		"model": model,
		"model_name": model_name
	}

	result = graph.invoke(initial_state)

	print(result)

if __name__ == "__main__":
	main()

