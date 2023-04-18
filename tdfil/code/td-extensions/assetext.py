"""
general asset extension class functions.
"""

import TDFunctions as TDF
from pathlib import Path
import json

class assetext:
	"""
	assetext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.opPanelCOMP = self.ownerComp.op('op_panel')
		self.thumbnailTOP = self.ownerComp.op('op_panel/thumbnail')

		self.getDAT = self.ownerComp.op('GET')
		self.setDAT = self.ownerComp.op('SET')
		self.bufferDAT = self.ownerComp.op('BUFFER')


	def Texture_Check(self, texture_file_parameter):
		'''
		Function for checking the existance of a texture file, and also checking it's extension is of an image.
		there's some extra logic here to fallback to green, if the path is entirely empty (a valid material can be be missing some textures)
		'''

		movie_types = [ f'.{x}' for x in tdu.fileTypes['movie'] ]
		image_types = [ f'.{x}' for x in tdu.fileTypes['image'] ]

		texture_exists_check = None
		if texture_file_parameter.eval() != '':
			if Path(texture_file_parameter.eval()).is_file():
				texture_exists_check = True
			else:
				texture_exists_check = False

		texture_extension_check = False
		if texture_exists_check == True:
			if Path(texture_file_parameter.eval()).suffix in image_types:
				texture_extension_check = True
		elif texture_exists_check == None:
			texture_extension_check = True
		elif texture_exists_check == False:
			extSplit = texture_file_parameter.eval().split('.')
			if len(extSplit) >= 2:
				if f'.{extSplit[-1]}' in image_types:
					texture_extension_check = True

		if texture_exists_check == None:
			texture_exists_check = True

		return { 'texture_exists_check':texture_exists_check, 'texture_extension_check':texture_extension_check }


	def ComputeBounds(self):
		
		Enableinstancing = self.ownerComp.par['Enableinstancing']
		Enableinstancing = False if Enableinstancing == None else Enableinstancing

		bounds = {
			'x_min':0,
			'y_min':0,
			'z_min':0,
			'x_max':0,
			'y_max':0,
			'z_max':0,
		}

		try:
			xformOnly_Padding = 0.1
			worldSpacePos = self.ownerComp.worldTransform * tdu.Position(0)
			bounds['x_min'] = worldSpacePos.x - xformOnly_Padding
			bounds['y_min'] = worldSpacePos.y - xformOnly_Padding
			bounds['z_min'] = worldSpacePos.z - xformOnly_Padding
			bounds['x_max'] = worldSpacePos.x + xformOnly_Padding
			bounds['y_max'] = worldSpacePos.y + xformOnly_Padding
			bounds['z_max'] = worldSpacePos.z + xformOnly_Padding
		except:
			debug(f'{self.ownerComp} is not a transform based COMP, cannot compute world space position, this item will be skipped...')
			bounds = None

		hasBounds = False

		### STANDARD GEO - computeBounds()
		if self.ownerComp.par.Objtype.eval() in op.TDFIL.Type_Group('GEOMETRY'):
			tdBounds = self.ownerComp.computeBounds()
			hasBounds = True

		### LIGHT COMP - computeBounds()
		elif self.ownerComp.par.Objtype.eval() in op.TDFIL.Type_Group('XFORM_LIGHTS'):
			tdBounds = self.ownerComp.op('geo_helper').computeBounds()
			hasBounds = True
		
		else:
			debug(f'computeBounds() not supported on: {self.ownerComp}')
			hasBounds = False
		
		if hasBounds == True and bounds != None:
			bounds['x_min'] = min( bounds['x_min'] , tdBounds.min.x )
			bounds['y_min'] = min( bounds['y_min'] , tdBounds.min.y )
			bounds['z_min'] = min( bounds['z_min'] , tdBounds.min.z )
			bounds['x_max'] = max( bounds['x_max'] , tdBounds.max.x )
			bounds['y_max'] = max( bounds['y_max'] , tdBounds.max.y )
			bounds['z_max'] = max( bounds['z_max'] , tdBounds.max.z )

		if Enableinstancing == True and bounds != None:
			instances = self.ownerComp.Get_All_Instances()
			instances = [ x for x in instances if x.selected == True ]
			for each in instances:
				instanceBounds = each.parent.obj.Compute_Instance_Bounds(each)
				bounds['x_min'] = min( instanceBounds['min'].x , bounds['x_min'] )
				bounds['y_min'] = min( instanceBounds['min'].y , bounds['y_min'] )
				bounds['z_min'] = min( instanceBounds['min'].z , bounds['z_min'] )
				bounds['x_max'] = max( instanceBounds['max'].x , bounds['x_max'] )
				bounds['y_max'] = max( instanceBounds['max'].y , bounds['y_max'] )
				bounds['z_max'] = max( instanceBounds['max'].z , bounds['z_max'] )

		return bounds
		# return tdBounds # why was I doing this??
	
	@property
	def Children_Immediate(self):
		immediateChildren = self.ownerComp.outputCOMPs
		return immediateChildren
	

	@property
	def Parent_Immediate(self):
		immediateParent = self.ownerComp.inputCOMPs
		return immediateParent[0] if len(immediateParent) > 0 else None
	
	@property
	def Children_Recursive(self):
		'''
		Psuedo recursively fetches all children of given parents, by the supplied number deep.
		default of 99 basically "guarantees" that we'll get all children cause who would have
		a scene hierarchy going that deep...
		'''

		levelsDeep = 99
		startingObjects = [self.ownerComp]

		foundChildren = []
		currentLevel = startingObjects
		for level in range(levelsDeep):
			nextLevel = []
			for object in currentLevel:
				
				found = [ x.id for x in op(object).Children_Immediate ]
				nextLevel += found
			
			currentLevel = nextLevel
			foundChildren += nextLevel
		
		foundChildren = [op(x) for x in foundChildren]
		
		return foundChildren
	
	@property
	def All_Parent_Generations(self):
		'''
		Psuedo recursively fetches all parent objects higher in the objects hierarchy.
		So, if this object is 3rd layer down parent hierarchy, this method will return the parent() and parent(2)
		in a sense.
		'''

		levelsDeep = 99
		startingObject = self.ownerComp

		foundParents = []
		currentLevel = startingObject
		for level in range(levelsDeep):
			
			Imediate_Parent = currentLevel.Parent_Immediate
			if Imediate_Parent != None:
				foundParents += [ Imediate_Parent.id ]
				currentLevel = Imediate_Parent
			else:
				break
		
		foundParents = [op(x) for x in foundParents]
		return foundParents

	def Material_Path(self):
		'''
		given a path to a file, this function tests it to see if it exists,
		and if it doesn't, returns a fallback path to a default geometry in 
		the td install directory. This helps prevent red meaningless errors.

		Additionally, this function also unloads the mesh, the the Loaded toggle is turned off.
		'''
		materialOP = self.ownerComp.par.Materialcomp.eval()
		return materialOP.op('mat') if materialOP != None else ''
	
	def Schedule_ThumbnailUpdate(self, delay_frames=0):
		'''
		Prime the thumbnail update by locking the thumbnail, and then calling the update.
		'''
		
		# to avoid multiple thumbnail updates from happening at the same time, we need to
		# check how many are currently queued, and then add that number to the delay_frames
		# so that the thumbnail update will happen after all the other queued ones.
		currently_qued_node_ids = [ x.group for x in runs if x.group != None ]
		num_currently_qued_nodes = len(list(set(currently_qued_node_ids)))	

		# op.TDFIL.TraceFunctionCall() # debugging

		# kill any existing runs for this node
		node_id = self.ownerComp.id
		for r in runs:
			if r.group == f'{node_id}':
				r.kill()
		
		# schedule a new run
		run(f'op({node_id}).Update_Thumbnail()', group=f'{node_id}', delayFrames=delay_frames + num_currently_qued_nodes)
		return

	def Update_Thumbnail(self):
		Thumbnailmanagercomp = None
		try:
			Thumbnailmanagercomp = parent.obj.par.Thumbnailmanagercomp.eval()
		except:
			pass
		
		try:
			Thumbnailmanagercomp = parent.Camera.par.Thumbnailmanagercomp.eval()
		except:
			pass
		
		if Thumbnailmanagercomp == None:
			debug('Thumbnailmanagercomp is invalid, skipping thumbnail generation...')
			return

		if self.thumbnailTOP == None:
			debug('thumbnail script top is invalid, skipping thumbnail generation...')
			return
		
		# thumbnails should be locked on startup, but in the case of a race condition, we lock it here anyways.
		if self.opPanelCOMP.op('thumbnail').lock == False:
			self.opPanelCOMP.op('thumbnail').lock = True
			return
		
		Thumbnailmanagercomp.Create_Thumbnail( self.thumbnailTOP , self.ownerComp , self.opPanelCOMP.width )

	def Get_Data(self):
		'''
		gets relevant scene data for this object. if the get/set/buffer dats are not present, we skip.

		'''
		if self.getDAT == None or self.bufferDAT == None or self.setDAT == None:
			debug(f'{self.ownerComp} has no get/set/buffer dats, cannot get data. early exit.')
			return

		self.getDAT.run()
		# self.setDAT
		data = json.loads(self.bufferDAT.text)

		return data


	def Set_Data(self, data):
		'''
		sets relevant scene data for this object. if the get/set/buffer dats are not present, we skip.
		
		'''
		if self.getDAT == None or self.bufferDAT == None or self.setDAT == None:
			debug(f'{self.ownerComp} has no get/set/buffer dats, cannot set data. early exit.')
			return

		self.bufferDAT.text = json.dumps(data, indent=4)
		self.setDAT.run()

		return

