import os

import google.genai as genai

from dotenv import load_dotenv
import ollama

from kg.neo4j_manager import Neo4jManager

load_dotenv()

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

if not neo4j_uri:
    raise ValueError("NEO4J_URI is missing. Set it to something like bolt://localhost:7687")

kg = Neo4jManager(
    uri=neo4j_uri,
    user=neo4j_user,
    password=neo4j_password
)

#proviamo neo4j
result = kg.query("MATCH (n) RETURN n LIMIT 5")
print(result)