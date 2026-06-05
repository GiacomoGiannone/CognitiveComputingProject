from neo4j import GraphDatabase


class Neo4jManager:

    def __init__(self, uri, user, password):

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )

    def query(self, cypher, params=None):

        with self.driver.session() as session:

            result = session.run(
                cypher,
                params or {}
            )

            return [r.data() for r in result]

    def close(self):
        self.driver.close()