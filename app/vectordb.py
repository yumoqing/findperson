from pymilvus import MilvusClient

class MilvusVectorDB:
    def __init__(self, dbname, tablename, dimension=768):
        self.dbname = dbname
        self.tablename = tablename
        self.client = MilvusClient(dbname)
        if not self.client.has_collection(collection_name=tablename):
            self.cient.create_collection(
                    collection_name=tablename,
                    dimension=dimension
            )

    def add(self, ns):
        return self.client.insert(collection_name=self.tablename, data=ns)

    def search_by_vector(self, vector, limit=5):
        return self.client.search(
                collection_name=self.tablename,
                data=[vector],
                limit=limit
        )

    def del(self, id):
        return self.client.delete(
                collection_name=self.tablename,
                ids=[id]
        )
