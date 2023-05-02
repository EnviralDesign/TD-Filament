

class sceneext:
	"""
	sceneext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
	
	@property
	def Camera(self):
		return self.ownerComp.par.Editorcomp.eval().Camera
	
	@property
	def SettingsManager(self):
		return self.ownerComp.par.Editorcomp.eval().SettingsManager
	
	@property
	def Objects(self):
		'''
		returns a list of objects that exist within the scene COMP, 
		that are at a depth of 1, and have a custom parameter called Objtype.
		'''
		comps = self.ownerComp.findChildren(parName='Objtype', depth=1)
		return comps
	
	@property
	def Instances(self):
		'''
		returns a list of instances that exist within any objects in the Scene COMP
		that have instancing enabled.
		'''
		
		comps = self.Objects
		comps = [ x for x in comps if x.par.Objtype.eval() in op.TDFIL.Type_Group('INSTANCEABLE') ] # mesh objects only.
		comps = [ x for x in comps if x.par.Enableinstancing.eval() == True ]
		
		instances = []
		for each in comps:
			instances += each.Get_All_Instances()
		return instances

	def Deselect_Special(self):
		'''
		There are some special objects like the camera that are stickied to the top of the outliner. the user can select these
		but they do not really belong to the general scene, they are above that. so, we have a special deselection function
		called here, that is called almost any time other selection functions are called so we don't get them intermingled.
		'''
		Cameracomp = self.Camera
		if Cameracomp != None:
			Cameracomp.selected = False
			Cameracomp.current = False

		Settingsmanagercomp = self.SettingsManager
		if Settingsmanagercomp != None:
			Settingsmanagercomp.selected = False
			Settingsmanagercomp.current = False

		return

	@property
	def Selected_Objects(self):
		return [ x for x in self.Objects if x.selected == True ]

	@property
	def Selected_Instances(self):
		return [ x for x in self.Instances if x.selected == True ]

	def Element_Type(self, object):
		return object.par.parentshortcut.eval()

	def Select_Objects(self, objects, clear_previous):
		if clear_previous == True:
			self.Deselect_All_Objects()
			self.Deselect_All_Instances()
		objects_marked_as_current = []
		for each in objects:
			self.Select_Object(object=each, clear_previous=False)
			if each.current == True:
				objects_marked_as_current += [each]
		
		for each in objects_marked_as_current:
			each.current = False
		if len(objects_marked_as_current) > 0:
			objects_marked_as_current[-1].current = True
		return

	def Select_Object(self, object, clear_previous):
		self.Deselect_Special()
		element_type = self.Element_Type(object)
		if element_type == 'obj':
			if clear_previous == True:
				self.Deselect_All_Objects()
				for each in self.Objects:
					if each.par.Objtype.eval() in op.TDFIL.Type_Group('INSTANCEABLE'):
						each.Deselect_All_Instances()
			object.selected = True
			object.current = True
		
		elif element_type == 'inst':
			if clear_previous == True:
				object.parent.obj.Deselect_All_Instances()
				self.Deselect_All_Objects()
			object.selected = True
			object.current = True
		return

	def Deselect_Object(self, object):
		self.Deselect_Special()
		element_type = self.Element_Type(object)
		if element_type == 'obj':
			object.selected = False
			object.current = False
		elif element_type == 'inst':
			object.selected = False
			object.current = False
			if len(object.parent.obj.Get_Selected_Instances()) == 0:
				object.parent.obj.selected = False
				object.parent.obj.current = False
		return
	
	def Deselect_Objects(self, objects):
		for each in objects:
			self.Deselect_Object(object=each)
		return

	def Deselect_All_Objects(self, exclusion=[]):
		self.Deselect_Special()
		for each in self.Objects:
			if each not in exclusion:
				each.selected = False
		return

	def Deselect_All_Instances(self, parent_object=None):
		self.Deselect_Special()
		if parent_object == None:
			instanceable_objects = [ x for x in self.Objects if x.par.Objtype.eval() in op.TDFIL.Type_Group('INSTANCEABLE') ]
			for each in instanceable_objects:
				each.Deselect_All_Instances()
		return

	def Duplicate_Selected(self):
		'''
		duplicates the currently selected objects or instances, selects the new ones, deselects everything else.
		'''

		self.Deselect_Special()

		Selected_Objects = self.Selected_Objects
		Selected_Instances = self.Selected_Instances

		# since objects can be duplicated in TD as a group with copyOPs, we can take advantage of this
		# but this command requires a shared parent, so we must split our selected items into groups pased
		# on common parent object.
		parent_groups = {}
		initial_selection_indicies = {}

		# group objects
		for each in Selected_Objects:
			group = parent_groups.get(each.parent().id,[])
			group += [ each.id ] + [ x.id for x in each.Children_Recursive ] # object but also recursive items.
			parent_groups[each.parent().id] = group

		# group instances
		for each in Selected_Instances:
			group = parent_groups.get(each.parent().id,[])
			group += [ each.id ]
			parent_groups[each.parent().id] = group

		
		# save selected items within each group
		for k,v in parent_groups.items():
			initial_selection_indicies[k] = [v.index(each) for each in v if op(each).selected == True]
		
		# iterate through groups, duplicating each and adding to our fresh list.
		newly_created_objects = []
		newly_to_select = []
		for parent,children in parent_groups.items():
			children = [ op(x) for x in children ]
			newly_created = op(parent).copyOPs(children)
			
			# offset new items below, and align X to be the same.
			for i,each in enumerate(newly_created):
				each.nodeY = children[i].nodeY - (each.nodeHeight * 1.1)
				each.nodeX = children[i].nodeX

				# carefully add the new duplicates who's position in their list matched the position of the previous selection.
				# this allows us to maintain the implied selection pattern the user started with, but on the newly duplicated items.
				if i in initial_selection_indicies[parent]:
					newly_to_select += [ each ]

			newly_created_objects += newly_created
		
		# deselect all old objects.
		self.Deselect_All_Objects()
		self.Deselect_All_Instances()

		for each in newly_to_select:
			op(each).selected = True
		return

	def Delete_Selected(self):
		'''
		Given a list of objects, this function deletes them. Currently no special care is taken for 
		objects who are parents other objects, this will not try and recursively delete children.
		'''

		self.Deselect_Special()

		Selected_Objects = self.Selected_Objects
		Selected_Instances = self.Selected_Instances

		# delete instances first..
		for each in Selected_Instances:
			# instances have no recursive children, so flat for loop is fine.
			each.destroy()
		
		for each in Selected_Objects:
			# objects can have hierarchical wire children, so we must destroy the children
			# first before we destroy the top level parent.
			for each2 in each.Children_Recursive:
				each2.destroy()
			each.destroy()

		return

	def Parent_Selected_To_Current(self):
		'''
		parents all selected items to the current item. which is usually just the last selected item.
		'''
		Selected_Instances = self.Selected_Instances
		if len(Selected_Instances) > 0:
			debug('cannot parent instances to other objects, aborting...')
			return
		
		Current_Object = self.ownerComp.currentChild
		Selected_Objects = [ x for x in self.Selected_Objects if x.id != Current_Object.id ]
		
		for each in Selected_Objects:
			each.inputCOMPConnectors[0].connect(Current_Object)

	def Unparent_Selected(self):
		'''
		Unparent the selected objects from their parents. This returns them to world space with no parents.
		'''
		Selected_Instances = self.Selected_Instances
		if len(Selected_Instances) > 0:
			debug('cannot unparent instances, aborting...')
			return
		
		Selected_Objects = self.Selected_Objects
		
		for each in Selected_Objects:
			each.inputCOMPConnectors[0].disconnect()


	def Select_Immediate_Parents(self):
		'''
		Selects the immediate parents one level up of all selected objects, replacing the previous selection.
		'''

		Selected_Instances = self.Selected_Instances
		if len(Selected_Instances) > 0:
			debug('This function is not designed for instances.. Please deselect any instances and try again.')
			return
		
		Selected_Objects = self.Selected_Objects
		newSelection = []
		for each in Selected_Objects:
			newSelection += each.inputCOMPs

		# if len(newSelection) == 0:
		# 	debug('there are no parents of the selected geometry, nothing to select. aborting...')
		# 	return
		
		self.Select_Objects(newSelection, clear_previous=True)
		return newSelection


	def Select_Immediate_Children(self):
		'''
		Selects the immediate children one level down of all selected objects, replacing the previous selection.
		'''

		Selected_Instances = self.Selected_Instances
		if len(Selected_Instances) > 0:
			debug('This function is not designed for instances.. Please deselect any instances and try again.')
			return
		
		Selected_Objects = self.Selected_Objects
		newSelection = []
		for each in Selected_Objects:
			newSelection += each.outputCOMPs

		# if len(newSelection) == 0:
		# 	debug('there are no parents of the selected geometry, nothing to select. aborting...')
		# 	return
		
		self.Select_Objects(newSelection, clear_previous=True)
		return newSelection


	def Delete_All(self):

		Objects = self.Objects
		Instances = self.Instances

		# delete instances first..
		for each in Instances:
			each.destroy()
		
		for each in Objects:
			each.destroy()