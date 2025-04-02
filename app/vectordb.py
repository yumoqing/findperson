from appPublic.jsonConfig import getConfig
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

class MilvusVectorDB:
    def __init__(self, orgid, dimension=768):
        config = getConfig()
        dbname = f'{config.vectordb_path}/{orgid}.db'
        self.orgid = orgid
        self.dbname = dbname
        self.dimension = dimension
        self.client = MilvusClient(dbname)

    def create_table_if_not_exists(self, tblname):
        if not self.client.has_collection(collection_name=tblname):
            fields = [
                    FieldSchema(name='id',dtype=DataType.VARCHAR,
                                    auto_id=False,
                                    is_primary=True, max_length=34),
                    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, 
                                        dim=self.dimension, description='vector'),
                    FieldSchema(name='imgid', dtype=DataType.VARCHAR, max_length=34),
                    FieldSchema(name='top', dtype=DataType.INT32),
                    FieldSchema(name='left', dtype=DataType.INT32),
                    FieldSchema(name='right', dtype=DataType.INT32),
                    FieldSchema(name='bottom', dtype=DataType.INT32),
                    FieldSchema(name='userid', dtype=DataType.VARCHAR, max_length=34)
            ]
            schema = CollectionSchema(fields=fields,auto_id=False, enable_dynamic_field=True)
            self.client.create_collection(
                    collection_name=tblname,
                    dimension=self.dimension,
                    schema=schema
            )

    def add(self, tblname, ns):
        self.create_table_if_not_exists(tblname)
        return self.client.insert(collection_name=tblname, data=ns)

    def search_by_vector(self, tblname, vector, limit=5):
        self.create_table_if_not_exists(tblname)
        return self.client.search(
                collection_name=tblname,
                data=[vector],
                limit=limit
        )

    def delete(self, tblname, id):
        self.create_table_if_not_exists(tblname)
        return self.client.delete(
                collection_name=tblname,
                ids=[id]
        )
