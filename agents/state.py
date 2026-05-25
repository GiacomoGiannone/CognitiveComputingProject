from typing import TypedDict, List, Any

class AgentState(TypedDict):
    #Struttura dati per rappresentare lo stato di un agente
    
    #cosa serve al planner agent?
    #riceve in input: richiesta utente, topics recenti del KG
    #torna in output: topic scelto

    user_input :str #il prompt o la richiesta dell'utente, da cui partire per scegliere un topic
    recent_topics :List[str] #lista di topic recenti estratti dal KG
    chosen_topic :str #topic scelto per approfondire, se applicabile

    #cosa serve al research agent?
    #riceve in input: topic scelto dal planner agent
    #torna in output: informazioni acquisite sul topic, da salvare nel KG

    verified_info :str #informazioni verificate e pronte per essere aggiunte al KG
    reasoning_trace :List[dict] #traccia del ragionamento usato per verificare le informazioni
    tool_outputs :dict #output dei tool usati (search, rag, kg, ecc.)

    #cosa serve al content creation agent?
    #riceve in input: informazioni verificate dal research agent
    #torna in output: contenuto creato (es. post, articolo) da aggiungere al KG

    created_content :str #contenuto creato pronto per essere aggiunto al KG

    #cosa serve al review agent?
    #riceve in input: contenuto creato dal content creation agent
    #torna in output: feedback sul contenuto, per migliorarlo prima di aggiungerlo al KG

    content_feedback :str #feedback sul contenuto creato, per migliorarlo

    #oggetti condivisi dal grafo
    kg :Any
    model :Any
    model_name :str



