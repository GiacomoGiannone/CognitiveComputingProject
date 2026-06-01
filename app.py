import google.genai as genai

from dotenv import load_dotenv
import ollama

from kg.KG import KG
from graph.graph import graph

def main():
	load_dotenv()
	
	#Inizializza il Knowledge Graph
	kg = KG()
	reset_kg = input("Vuoi resettare il KG? (y/n): ").strip().lower()
	if reset_kg == "y":
		kg.reset() #Pulisce il grafo all'inizio di ogni esecuzione, per testare da zero

	#Inizializza il modello (sostituisci con il tuo modello preferito)
	model = ollama.Client()
	model_name = "llama3.1"

	use_suggestion = input("Vuoi un suggerimento di argomento? (y/n): ").strip().lower()
	if use_suggestion == "y":
		planner_mode = "suggest"
		prompt = "Suggerisci un topic per un blog di calcio europeo."
	else:
		planner_mode = "choose"
		prompt = input("Inserisci il prompt per il modello: ")

	initial_state = {
		"user_input": prompt,
		"recent_topics": [],
		"chosen_topic": None,
		"suggested_topic": None,
		"planner_mode": planner_mode,
		"verified_info": None,
		"reasoning_trace": [],
		"tool_outputs": {},
		"created_content": None,
		"content_feedback": None,
		"content_feedback_detail": "",
		"kg": kg,
		"model": model,
		"model_name": model_name
	}

	result = graph.invoke(initial_state)

if __name__ == "__main__":
	main()

