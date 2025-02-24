import pickle

def save_file(obj,filepath):
	with open(filepath,'wb') as out:
		pickle.dump(obj,out,pickle.HIGHEST_PROTOCOL)

def load_file(filepath):
	with open(filepath, 'rb') as inp:
		obj=pickle.load(inp)
	return obj

if __name__=='__main__':
	pass