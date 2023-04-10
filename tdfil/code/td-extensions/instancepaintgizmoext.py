# import TDFunctions as TDF

import random

class instancepaintgizmoext:
	"""
	instancepaintgizmoext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.sceneObjectListDAT = self.ownerComp.op(op('null_scene_object_list'))

		self.lastPosition = tdu.Position(0)
		self.accumulatedDistance = 0
		self.selectedItems = []
		self.itemOffsets = {}
		self.sharedParent = None

		self.Brush_UpdateSelection() # updates selectedItems internally.

	############################
	### PRIVATE FUNCTIONS
	############################

	def update_brush_helper( self , pos , nrm ):
		
		Brushmode = self.ownerComp.par.Brushmode.eval()

		m = tdu.Matrix()

		if Brushmode == 'pos':
			m.translate( pos )
		
		elif Brushmode == 'posnrm':
			q = tdu.Quaternion(tdu.Vector(0, 1, 0), tdu.Vector(nrm))
			rot = q.eulerAngles(order='xyz')
			m.rotate(rot)
			m.translate( pos )
		
		s,r,t = m.decompose()
		parent.obj.par.tx = t[0]
		parent.obj.par.ty = t[1]
		parent.obj.par.tz = t[2]
		parent.obj.par.rx = r[0]
		parent.obj.par.ry = r[1]
		parent.obj.par.rz = r[2]


	############################
	### PUBLIC FUNCTIONS
	############################

	def Set_Mode(self, modeString):
		'''	
		modeString = ['select','pos','posnrm']
		'''
		currentModeString = self.ownerComp.par.Brushmode.eval()
		newModeString = modeString.lower()
		self.ownerComp.par.Brushmode = newModeString

		return

	@property
	def Picking_Active(self):
		return self.ownerComp.par.Brushmode.eval() != 'select'

	def Create_Payload(self, pos, nrm, uvw, col):
		'''
		generate the payload. This is a big function and an important one. 
		It takes in data from the render pick, position, normal, uvw, color, 
		and using the parent gizmo's custom parameters, jitters the instances being painted
		using that information.
		'''
		payload = {}

		######################### TRANSLATION ##########################################

		# normal offset scaling.
		Normaloffset = self.ownerComp.par.Normaloffset.eval()

		# we want the brushes up vector, but as it points in world space.
		brush_ws_y_vector = self.ownerComp.worldTransform * tdu.Vector(0,1,0)
		brush_ws_y_vector.normalize()

		if self.ownerComp.par.Brushradius > 0: # if brush has any radius at all,we need to jitter spatially.

			# we want the X vector of the brush, but ias it points in world space.
			# NOTE We could also use Z vector in world space, as it also is aligned to the same plane.
			brush_ws_x_vector = self.ownerComp.worldTransform * tdu.Vector(1,0,0)
			brush_ws_x_vector.normalize()

			# now we scale the X vector by radius and then add randomization to it.
			brush_ws_x_vector *= self.ownerComp.par.Brushradius * random.uniform(0, 1)

			# now rotate the vector around the brush up vector randomly.
			# this is polar plane aligned randomization. ensures our painted instance falls within our brush radius.
			m = tdu.Matrix(); m.rotateOnAxis(brush_ws_y_vector , random.uniform(0, 360))
			brush_ws_x_vector = m * brush_ws_x_vector

			# add the jitter offset to the world space position of the brush cursor.
			payload['tx'] = pos[0] + brush_ws_x_vector.x + ( brush_ws_y_vector.x * Normaloffset )
			payload['ty'] = pos[1] + brush_ws_x_vector.y + ( brush_ws_y_vector.y * Normaloffset )
			payload['tz'] = pos[2] + brush_ws_x_vector.z + ( brush_ws_y_vector.z * Normaloffset )
		
		else: # if the brush radius is zero, we can save ourselves some calculation since jitter will also be zero.
			
			payload['tx'] = pos[0] + ( brush_ws_y_vector.x * Normaloffset )
			payload['ty'] = pos[1] + ( brush_ws_y_vector.y * Normaloffset )
			payload['tz'] = pos[2] + ( brush_ws_y_vector.z * Normaloffset )

		######################### ROTATION ##########################################
		Rotationspace = self.ownerComp.par.Rotationspace.eval()
		Rotationjitterx = self.ownerComp.par.Rotationjitterx.eval()
		Rotationjittery = self.ownerComp.par.Rotationjittery.eval()
		Rotationjitterz = self.ownerComp.par.Rotationjitterz.eval()

		if Rotationspace == 'world': # if world space, we are jittering the instances rotation values directly.
			rx = random.uniform(-Rotationjitterx,Rotationjitterx)
			ry = random.uniform(-Rotationjittery,Rotationjittery)
			rz = random.uniform(-Rotationjitterz,Rotationjitterz)
		
		elif Rotationspace == 'brush': # if brush space, we want to jitter the rotation, but within brush space.
			rx = random.uniform(-Rotationjitterx,Rotationjitterx)
			ry = random.uniform(-Rotationjittery,Rotationjittery)
			rz = random.uniform(-Rotationjitterz,Rotationjitterz)
			m = tdu.Matrix()
			m.rotate(rx,ry,rz)
			r = self.ownerComp.worldTransform.decompose()[1]
			m.rotate(r)
			r = m.decompose()[1]
			rx = r[0]
			ry = r[1]
			rz = r[2]

		payload['rx'],payload['ry'],payload['rz'] = rx,ry,rz

		######################### SCALE ##########################################
		Scalejittermode = self.ownerComp.par.Scalejittermode.eval()
		Scalejitteruniform = self.ownerComp.par.Scalejitteruniform.eval()
		Scalejitterseparatex = self.ownerComp.par.Scalejitterseparatex.eval()
		Scalejitterseparatey = self.ownerComp.par.Scalejitterseparatey.eval()
		Scalejitterseparatez = self.ownerComp.par.Scalejitterseparatez.eval()
		if Scalejittermode == 'uniform':
			scalejitter = random.uniform(1-Scalejitteruniform , 1+Scalejitteruniform)
			Scalejitterx = scalejitter
			Scalejittery = scalejitter
			Scalejitterz = scalejitter
		elif Scalejittermode == 'separate':
			Scalejitterx = random.uniform(1-Scalejitterseparatex , 1+Scalejitterseparatex)
			Scalejittery = random.uniform(1-Scalejitterseparatey , 1+Scalejitterseparatey)
			Scalejitterz = random.uniform(1-Scalejitterseparatez , 1+Scalejitterseparatez)

		payload['sx'],payload['sy'],payload['sz'] = Scalejitterx,Scalejittery,Scalejitterz

		######################### COLOR ##########################################
		Colorjitterr = self.ownerComp.par.Colorjitterr.eval()
		Colorjitterg = self.ownerComp.par.Colorjitterg.eval()
		Colorjitterb = self.ownerComp.par.Colorjitterb.eval()
		Colorjittera = self.ownerComp.par.Colorjittera.eval()
		Colorjitteruniformity = self.ownerComp.par.Colorjitteruniformity.eval()
		isThereColorJitter = max([Colorjitterr,Colorjitterg,Colorjitterb,Colorjittera]) > 0

		if isThereColorJitter == True:

			r = random.uniform(1-Colorjitterr,1)
			g = random.uniform(1-Colorjitterg,1)
			b = random.uniform(1-Colorjitterb,1)
			a = random.uniform(1-Colorjittera,1)
			lum = self.ownerComp.Rgb_2_Luminance(r,g,b)

			# based on the color uniformity, we fade percentage wise, to the luminance of the jittered color
			# if luminance uniformity is zero, this has no effect.
			r = self.ownerComp.Mix( r , lum , Colorjitteruniformity )
			g = self.ownerComp.Mix( g , lum , Colorjitteruniformity )
			b = self.ownerComp.Mix( b , lum , Colorjitteruniformity )
			
		else:
			r,g,b,a = 1,1,1,1
		
		payload['r'] = r * col[0]
		payload['g'] = g * col[1]
		payload['b'] = b * col[2]
		payload['a'] = a * col[3]

		######################### UVS ##########################################
		Uvwoffsetjitteru = self.ownerComp.par.Uvwoffsetjitteru.eval()
		Uvwoffsetjitterv = self.ownerComp.par.Uvwoffsetjitterv.eval()
		Uvwoffsetjitterw = self.ownerComp.par.Uvwoffsetjitterw.eval()
		isThereUvJitter = max([Uvwoffsetjitteru,Uvwoffsetjitterv,Uvwoffsetjitterw]) > 0

		if isThereUvJitter == True:

			payload['u'] = random.uniform( -Uvwoffsetjitteru , Uvwoffsetjitteru )
			payload['v'] = random.uniform( -Uvwoffsetjitterv , Uvwoffsetjitterv )
			payload['w'] = random.uniform( -Uvwoffsetjitterw , Uvwoffsetjitterw )

		return payload


	def Brush_UpdateSelection(self):

		self.itemOffsets = {}

		instanceable_obj_types = op.TDFIL.Type_Group('INSTANCEABLE')
		null_obj_types = op.TDFIL.Type_Group('NULL')

		self.sceneObjectListDAT.cook(force=True)
		paths = self.sceneObjectListDAT.col('path')[1::] # get path column, but omit header.
		selected = self.sceneObjectListDAT.col('selected')[1::] # get selected column, but omit header.
		selected = [ x.row for x in selected if x.val == '1'] # get rows that have selection status of 1
		selected = [ op(x) for x in paths if x.row in selected] # get operators that are in selected rows only.
		selected = [ x for x in selected if x.par.Objtype.eval() in instanceable_obj_types + null_obj_types ] # prune down to only objects that can be instanced.

		# instance painting of more than one object not allowed. this helps enforce the user to instance paint single objects OR instance paint nulls with multiple items parented to it.
		if len(selected) > 1:
			debug('cannot instance paint more than one object at a time. If you wish to instance collections of items, parent them to a null, and instance paint the null..')
			self.selectedItems = []
			self.sharedParent = None
			return self.selectedItems
		
		elif len(selected) == 1:
			if selected[0].par.Objtype.eval() in null_obj_types:
				self.sharedParent = selected[0]
			else:
				self.sharedParent = None
		
		else:
			self.sharedParent = None
		
		# build secondary reecursive children dict and list.
		selected_secondary = {}
		secondary = []
		for each in selected:
			Children_Recursive = each.Children_Recursive
			Children_Recursive = [ x for x in Children_Recursive if x not in selected ] # don't include recurse children if already selected.
			Children_Recursive = [ x for x in Children_Recursive if x.par.Objtype.eval() in instanceable_obj_types ] # only include instanceable types.
			
			selected_secondary[each.id] = Children_Recursive
			secondary += Children_Recursive
		
		# iterate through our recursive parent hierarchy
		for recursiveParent, recursiveChildren in selected_secondary.items():
			
			# if parent is not instance enabled, enable it.
			recursiveParent = op(recursiveParent)
			if recursiveParent.par['Enableinstancing'] != None:
				if recursiveParent.par.Enableinstancing == False:
					debug(f'Enabling Instancing for {recursiveParent.name}')
					recursiveParent.par.Enableinstancing = True

			# now iterate through top parent's recursive children.	
			for child in recursiveChildren:

				# if not instance enabled, enable it.
				if child.par.Enableinstancing == False:
					debug(f'Enabling Instancing for {child.name}')
					child.par.Enableinstancing = True

				# write out transform data relative to child.
				m = recursiveParent.relativeTransform(child)
				s,r,t = m.decompose()
				self.itemOffsets[child.id] = {
					'tx':t[0],'ty':t[1],'tz':t[2],
					'rx':r[0],'ry':r[1],'rz':r[2],
					'sx':s[0],'sy':s[1],'sz':s[2],
				}

		self.selectedItems = selected + secondary
		return self.selectedItems

	def Brush_SetPickingOnActive(self, state):
		for each in self.selectedItems:
			each.pickable = state
			try:
				each.op('geo_icon').pickable = state
			except:
				pass
			try:
				each.op('geo_mesh').pickable = state
			except:
				pass

	def Brush_Init(self, pos , nrm , uvw , col):
		'''
		updates the cursor's position and orientation via render pick geometry position/normal.
		'''
		self.update_brush_helper(pos,nrm)
		self.lastPosition.x = pos[0]
		self.lastPosition.y = pos[1]
		self.lastPosition.z = pos[2]
		self.accumulatedDistance = 9999999
		
		return

	def Brush_Paint(self, pos, nrm, uvw, col):

		if len(self.selectedItems) == 0:
			debug('no instancable objects selected. Please check selection, and ensure instancing is enabled on your objects.')
			return

		self.update_brush_helper(pos,nrm)
		self.accumulatedDistance += (self.lastPosition - tdu.Position(pos)).length()

		if self.accumulatedDistance > parent.obj.par.Brushdrawspacing.eval():
			payload = self.Create_Payload(pos,nrm,uvw,col)
			for each in self.selectedItems:
				secondaryOffsets = self.itemOffsets.get(each.id,None)
				# print( self.sharedParent )
				each.Create_Instance( payload, secondaryOffsets, self.sharedParent)
			self.accumulatedDistance = 0
			self.lastPosition = tdu.Position(pos)

		return

	def Brush_Erase(self, pos, nrm):
		if len(self.selectedItems) == 0:
			debug('no instancable objects selected. Please check selection, and ensure instancing is enabled on your objects.')
			return

		self.update_brush_helper(pos,nrm)
		self.accumulatedDistance = 0
		self.lastPosition = tdu.Position(pos)
		payload = {
			'Brushradius':parent.obj.par.Brushradius.eval(),
			'Brushx':pos[0],
			'Brushy':pos[1],
			'Brushz':pos[2],
		}

		for each in self.selectedItems:
			each.Delete_Instance( payload )

		return

	def Brush_End(self, pos, nrm, uvw, col):
		
		instanceable_obj_types = parent.Viewport.Type_Group('INSTANCEABLE')

		if len(self.selectedItems) == 0:
			debug('no instancable objects selected. Please check selection, and ensure instancing is enabled on your objects.')

		self.update_brush_helper(pos,nrm)
		self.accumulatedDistance = 0

		for each in self.selectedItems:
			if each.par.Objtype.eval() in instanceable_obj_types:
				each.Layout_Instances()


		return