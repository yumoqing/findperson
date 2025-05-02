from ahserver.webapp import webapp
from ahserver.serverenv import ServerEnv
from appPublic.worker import awaitify
from findperson.init import load_findperson

def get_module_dbname(name):
	return 'imagefind'

def init():
	g = ServerEnv()
	g.get_module_dbname = get_module_dbname
	load_findperson

if __name__ == '__main__':
	webapp(init)
