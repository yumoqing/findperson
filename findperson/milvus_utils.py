from appPublic.jsonConfig import getConfig
from appPublic.dictObject import DictObject
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

class MilvusCollection:
	def __init__(self, client, name, dimension):
		self.client = client
		self.name = name,
		self.dimension = dimension
		sellf.set_fields()
		self.set_indexes()
		self.vectorfield=None
		self.output_fields = []
	
	def set_fields(self):
		self.fields = []
	
	def set_indexes():
		self.indexes = []

	def create_vector_index(self):
		index_params = self.client.prepare_index_params()
		for idx in self.indexes:
			index_params.add_index(**idx)
		self.client.create_index(self.name, index_params)

	def create_table_if_not_exists(self):
		if not self.client.has_collection(collection_name=self.name):
			fields = []
			for f in self.fields:
				f = DictObject(**f)
				if f.dtype == DataType.FLOAT_VECTOR:
					self.vectorfield = f.name
					f.dim=self.dimension
				else:
					self.output_fields.append(f.name)
				fields.append(FieldSchema(**f))
			schema = CollectionSchema(fields=fields,auto_id=False, enable_dynamic_field=True)
			self.client.create_collection(
					collection_name=self.name,
					dimension=self.dimension,
					schema=schema
			)

    def add(self, ns,flush=False):
        self.create_table_if_not_exists(self.name)
        ret = self.client.insert(collection_name=self.name, data=ns)
        if flush:
            self.create_vector_index(self.name)
        return ret

    def search_by_vector(self, vector, limit=5):
        self.create_table_if_not_exists(self.name)
        return self.client.search(
                collection_name=self.name,
                anns_field=self.vectorfield,
                data=[vector],
                output_fields=self.output_fields,
                limit=limit
        )

    def delete(self, id):
        self.create_table_if_not_exists(self.name)
        return self.client.delete(
                collection_name=self.name,
                ids=[id]
        )

class MilvusVectorDB:
	def __init__(self, dimension=128):
		config = getConfig()
		dbname = config.vectordb_path
		self.dbname = dbname
		self.dimension = dimension
		self.client = MilvusClient(dbname)

class Faces(MilvusCollection):
	def set_fields(self):
		self.fields = [
			dict(name='id',dtype=DataType.VARCHAR,
						auto_id=False,
						is_primary=True, max_length=34),
			dict(name='vector', dtype=DataType.FLOAT_VECTOR, 
						dim=self.dimension, description='vector'),
			dict(name='imgid', dtype=DataType.VARCHAR, max_length=34),
			dict(name='imagepath', dtype=DataType.VARCHAR, max_length=500),
			dict(name='top', dtype=DataType.INT32),
			dict(name='left', dtype=DataType.INT32),
			dict(name='right', dtype=DataType.INT32),
			dict(name='bottom', dtype=DataType.INT32),
			dict(name='userid', dtype=DataType.VARCHAR, max_length=34)
		]
	def set_indexs(self):
		self.indexes = {
			"index_type": "IVF_FLAT",  # Choose index type
			"field_name":"vector",
			"index_name":"vector_index",
			# Distance metric: L2 (Euclidean) or IP (Inner Product)
			"metric_type": "L2",  
			"params": {"nlist": 128}  # Number of clusters for IVF
		}

	def create_vector_index(self, tblname):
		self.client.flush(tblname)
		index_params = self.client.prepare_index_params()
		index_params.add_index(**{
			"index_type": "IVF_FLAT",  # Choose index type
			"field_name":"vector",
			"index_name":"vector_index",
			"metric_type": "L2",  # Distance metric: L2 (Euclidean) or IP (Inner Product)
			"params": {"nlist": 128}  # Number of clusters for IVF
		})
		self.client.create_index(tblname, index_params)
		print("index created")

	def create_table_if_not_exists(self, tblname):
		if not self.client.has_collection(collection_name=tblname):
			fields = [
					FieldSchema(name='id',dtype=DataType.VARCHAR,
									auto_id=False,
									is_primary=True, max_length=34),
					FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, 
										dim=self.dimension, description='vector'),
					FieldSchema(name='imgid', dtype=DataType.VARCHAR, max_length=34),
					FieldSchema(name='imagepath', dtype=DataType.VARCHAR, max_length=500),
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

	def add(self, tblname, ns,flush=False):
		self.create_table_if_not_exists(tblname)
		ret = self.client.insert(collection_name=tblname, data=ns)
		if flush:
			self.create_vector_index(tblname)
		return ret

	def search_by_vector(self, tblname, vector, limit=5):
		self.create_table_if_not_exists(tblname)
		return self.client.search(
				collection_name=tblname,
				anns_field="vector",
				data=[vector],
				output_fields=["imgid", "imagepath", "top","right","bottom", "left", "userid" ],
				limit=limit
		)

	def delete(self, tblname, id):
		self.create_table_if_not_exists(tblname)
		return self.client.delete(
				collection_name=tblname,
				ids=[id]
		)
