"""
mesh specific asset description goes here
"""

import TDFunctions as TDF
from pathlib import Path
import collections
import math

class meshext:
	"""
	meshext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.opPanelCOMP = self.ownerComp.op('op_panel')
		self.thumbnailTOP = self.ownerComp.op('op_panel/thumbnail')
		self.instancerCOMP = self.ownerComp.op('base_instancer')
		self.numInstancesCHOP = self.ownerComp.op('null_num_instances')
		self.instanceCHOP = self.ownerComp.op('null_inst')
		# self.instanceListDAT = self.instancerCOMP.op('null_instance_data') if self.instancerCOMP != None else None


	def Init_Instance(self):
		'''
		Inits a single instance if no instance leafs exist, within this object.
		'''

		f = self.instancerCOMP.findChildren(type=geometryCOMP, depth=1)
		if len(f) == 0:
			template = ipar.Viewport.Instancetemplatecomp.eval()
			n = self.instancerCOMP.copy(template)
			n.nodeX = 0; n.nodeY = 0
			n.name += '_0'

	def Create_Instance(self, payload, offsets, sharedParent):


		if parent.obj.par.Objtype.eval() not in parent.Viewport.Type_Group('INSTANCEABLE'):
			# debug(f'cannot create instance of object type {parent.obj.par.Objtype}, this is an error, you shouldnt have gotten this far...')
			# disabling this error, because if the user has some items parented to a null, we want them to be able to instance the children of the null as a group.
			pass
		else:
			template = ipar.Viewport.Instancetemplatecomp.eval()
			n = self.instancerCOMP.copy(template)

			tx,ty,tz = payload.get('tx',0),payload.get('ty',0),payload.get('tz',0)
			rx,ry,rz = payload.get('rx',0),payload.get('ry',0),payload.get('rz',0)
			sx,sy,sz = payload.get('sx',1),payload.get('sy',1),payload.get('sz',1)
			r,g,b,a  = payload.get('r',1),payload.get('g',1),payload.get('b',1),payload.get('a',1)
			u,v,w  = payload.get('u',1),payload.get('v',1),payload.get('w',1)

			if isinstance(offsets,dict):
				txo,tyo,tzo = offsets.get('tx',0),offsets.get('ty',0),offsets.get('tz',0)
				rxo,ryo,rzo = offsets.get('rx',0),offsets.get('ry',0),offsets.get('rz',0)
				sxo,syo,szo = offsets.get('sx',1),offsets.get('sy',1),offsets.get('sz',1)
				offset_matrix = tdu.Matrix()
				offset_matrix.scale(sxo,syo,szo)
				offset_matrix.rotate(rxo,ryo,rzo)
				offset_matrix.translate(txo,tyo,tzo)
			else:
				offset_matrix = tdu.Matrix()

			if sharedParent == None:
				parent_matrix_inverse = self.ownerComp.worldTransform
				parent_matrix_inverse.invert()
			else:
				parent_matrix_inverse = self.ownerComp.relativeTransform(sharedParent)
				# parent_matrix_inverse = tdu.Matrix()
				# parent_matrix_inverse.rotate(0,180,0)
				# parent_matrix_inverse = sharedParent.relativeTransform(self.ownerComp)
				# parent_matrix_inverse = sharedParent.worldTransform
				# parent_matrix_inverse.invert()
				# print('ASKLJHDSDEFGHKSFDGNMB<')
				# parent_matrix_inverse.invert()
			
			ttt = parent_matrix_inverse.decompose()[2]
			rrr = parent_matrix_inverse.decompose()[1]
			sss = parent_matrix_inverse.decompose()[0]
			# print(self.ownerComp, rrr)
			# print(rx,ry,rz)

			instance_matrix = tdu.Matrix()
			instance_matrix.scale(sx,sy,sz)
			instance_matrix.rotate(rx,ry,rz)
			instance_matrix.translate(tx,ty,tz)

			final_matrix = parent_matrix_inverse * instance_matrix * offset_matrix
			# final_matrix = parent_matrix_inverse * instance_matrix

			s_,r_,t_ = final_matrix.decompose()

			# we use the raw rotation and scale values, but use the position values derived
			# from the relative position that takes parent transforms into account.
			# this basically means the instance will always be painted "where we click" even if parent is transformed/scaled/rotated.
			n.par.tx,n.par.ty,n.par.tz = t_
			n.par.rx,n.par.ry,n.par.rz = r_
			n.par.sx,n.par.sy,n.par.sz = s_

			n.par.Colormultiplyr = r
			n.par.Colormultiplyg = g
			n.par.Colormultiplyb = b
			n.par.Colormultiplya = a

			n.par.Uvoffsetu = u
			n.par.Uvoffsetv = v
			n.par.Uvoffsetw = w
		
		return

	def Delete_Instance(self, payload):

		Brushradius = payload['Brushradius']
		if Brushradius == 0:
			debug('cannot delete instances based on brush size, because brush size is == 0. Increase Brush size and try again.')
			return

		if parent.obj.par.Objtype.eval() not in parent.Viewport.Type_Group('INSTANCEABLE'):
			# debug('some of the selection does not support instancing... skipping those.')
			# disabling this error, because if the user has some items parented to a null, we want them to be able to instance the children of the null as a group.
			return
		
		brushpos = tdu.Position(payload['Brushx'],payload['Brushy'],payload['Brushz'])
		
		instances = self.Get_All_Instances()
		for each in instances:
			instancepos = tdu.Position(each.worldTransform.decompose()[2])
			distance = (instancepos - brushpos).length()
			
			if distance <= Brushradius:
				each.destroy()
		
		return


	def Layout_Instances(self):
		'''
		attempts to relayout the instance COMP's into a grid formation.
		'''
		f = self.instancerCOMP.findChildren(type=geometryCOMP, depth=1)
		rowcol = int(math.sqrt(len(f))+1)
		spacing = 200

		for i,each in enumerate(f):
			x = i % rowcol
			y = int(i / rowcol)
			each.nodeX = (x * spacing)
			each.nodeY = (y * -spacing) - each.nodeHeight
		
		return

	def Delete_All_Instances(self):
		'''
		Deletes all the instances present inside the object, and disables instancing.
		'''

		parent.obj.par.Enableinstancing = False

		f = self.instancerCOMP.findChildren(type=geometryCOMP, depth=1)
		for each in f:
			each.destroy()


	def Get_Instance(self, instanceId):
		'''
		given an instance ID, returns a reference to the sub instance operator.
		this involves a bit of translation/lookup since it's possible the instances are being
		sorted cpu side before going to GPU.

		NOTE; this works because an important assumption
		join chops when using pattern matching aggregate a list of operators in the same fashion that findChildren() does.
		hence, we can use the index channel to directly pull from the findChildren() results.
		'''

		originalInstanceID = op(int(self.instanceCHOP['opid'][instanceId]))
		return originalInstanceID


	def Num_Instances(self):
		'''
		just gets the number of instances this object has.
		'''

		self.numInstancesCHOP.cook(force=True)
		numInstances = int(self.numInstancesCHOP[0])

		return numInstances


	def Get_All_Instances(self):

		f = self.instancerCOMP.findChildren(type=geometryCOMP, depth=1)

		return f


	def Get_Selected_Instances(self):

		f = self.instancerCOMP.findChildren(type=geometryCOMP, depth=1)
		f = [ x for x in f if x.selected == True ]

		return f


	def Deselect_All_Instances(self):

		f = self.instancerCOMP.findChildren(type=geometryCOMP, depth=1)
		for each in f:
			each.selected = False
			each.current = False

		return f

	def Compute_Instance_Bounds(self, instanceOP):

		sop_min = self.ownerComp.op('geo_mesh/in1').min
		sop_max = self.ownerComp.op('geo_mesh/in1').max
		
		sop_min = instanceOP.worldTransform * sop_min
		sop_max = instanceOP.worldTransform * sop_max
		
		sop_min.x = min(sop_min.x , sop_max.x)
		sop_min.y = min(sop_min.y , sop_max.y)
		sop_min.z = min(sop_min.z , sop_max.z)
		
		sop_max.x = max(sop_min.x , sop_max.x)
		sop_max.y = max(sop_min.y , sop_max.y)
		sop_max.z = max(sop_min.z , sop_max.z)

		return {'min':sop_min, 'max':sop_max}

