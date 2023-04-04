import TDFunctions as TDF
# from TDStoreTools import StorageManager

class Outliner:
	"""
	Outliner description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.sceneListDAT = op('null_scene_list')

		# properties
		TDF.createProperty(self, 'Memory', value={'Collapsed':[]}, dependable='deep',
						   readOnly=False)

	def ExpandCollapseToggle(self, keys):
		for key in keys:
			if key not in self.Memory['Collapsed']:
				self.Memory['Collapsed'] += [key]
			else:
				self.Memory['Collapsed'] = [ x for x in self.Memory['Collapsed'] if x != key ]
		return

	def Collapse(self, keys):
		for key in keys:
			if key not in self.Memory['Collapsed']:
				self.Memory['Collapsed'] += [key]
		return
	
	def Expand(self, keys):
		for key in keys:
			self.Memory['Collapsed'] = [ x for x in self.Memory['Collapsed'] if x != key ]
		return
	
	def ExpandAll(self):
		'''
		expands all the collapsed object hierarchies to be visible.
		'''
		self.Memory['Collapsed'] = []
		return
	
	def CollapseAll(self):
		'''
		collapses all the object hierarchies that contain children to be hidden.
		This works by going through the scene list dat, finding all items that have any children
		then for each of those aquiring the operator's ID, and putting that in the Collapsed list.
		'''
		ids = list(map(int,self.sceneListDAT.col('id')[1::]))
		children = list(map(int,self.sceneListDAT.col('Numchildren')[1::]))
		instances = list(map(int,self.sceneListDAT.col('Numinstances')[1::]))
		subitems = [ ids[i] for i,each in enumerate(zip(children,instances)) if max(each) > 0]
		self.Memory['Collapsed'] = subitems
		return

	@property
	def Collapsed(self):
		return self.Memory['Collapsed']

	@property
	def AnyCollapsed(self):
		return len( self.Memory['Collapsed'] ) > 0
	
	def Is_Collapsed(self, idPath):
		'''
		Given an ID path, returns if that particular object is visible or not by virtue
		of any parents in it's hierarchy not being visible.
		'''
		ret = False

		for key in self.Memory['Collapsed']:
			keystr = str(key)
			result = idPath.find( keystr )

			if result == -1:
				pass
			
			else:
				keysubstr = idPath[result:result+key]
				keysubstr = keysubstr.replace('/','')
				
				match = len(keysubstr) != len(keystr)
				ret = max( ret , match )

		return ret