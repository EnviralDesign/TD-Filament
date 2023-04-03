import math

class CameraUtilsExt:
	"""
	CameraUtilsExt description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

	def Generic_Bounds_Fallback(self, object):
		'''
		Objects should have an extension that has the function called ComputeBounds()
		If they do not have it, then we can use this function to generically get the world space
		position of any 3d COMP that has a transform matrix in TD.
		'''
		worldSpacePos = object.worldTransform * tdu.Position(0,0,0)
		bounds = {
			'x_min':worldSpacePos.x,
			'y_min':worldSpacePos.y,
			'z_min':worldSpacePos.z,
			'x_max':worldSpacePos.x,
			'y_max':worldSpacePos.y,
			'z_max':worldSpacePos.z,
		}
		return bounds

	def distance_3d(self, x1,y1,z1, x2,y2,z2):
		return math.sqrt( ((x2-x1)**2) + ((y2-y1)**2) + ((z2-z1)**2) )


	def Home_Camera_To_Selected_Objects(self):
		# debug( 'homing cam to...' )


		CameraCOMP = ipar.Viewport.Cameracomp.eval()
		ViewMatrix = CameraCOMP.worldTransform
		ViewDirection = ViewMatrix * tdu.Vector(0,0,-1)
		ViewDirection.normalize()
		SceneCOMP = ipar.Viewport.Scenecomp.eval()
		scene_objects = SceneCOMP.Objects
		scene_objects = [ x for x in scene_objects if x.selected == True ]

		min_x_ = []
		min_y_ = []
		min_z_ = []
		max_x_ = []
		max_y_ = []
		max_z_ = []

		for each in scene_objects:
			
			if hasattr( each , 'ComputeBounds' ):
				b = each.ComputeBounds()
			else:
				debug(f'ComputeBounds() did not exist on {each}, falling back to generic bounds function.')
				b = self.Generic_Bounds_Fallback(each)

			if b != None: # things like material comps return None, since they do not have spatial transforms.
				min_x_ += [ b['x_min'] ]
				min_y_ += [ b['y_min'] ]
				min_z_ += [ b['z_min'] ]
				max_x_ += [ b['x_max'] ]
				max_y_ += [ b['y_max'] ]
				max_z_ += [ b['z_max'] ]
		
		if len(min_x_) == 0:
			debug('no items were selected that could be homed to... skipping.')
			return

		min_x = min(min_x_)
		min_y = min(min_y_)
		min_z = min(min_z_)
		max_x = max(max_x_)
		max_y = max(max_y_)
		max_z = max(max_z_)

		boundingSphere_x = (max_x + min_x) / 2
		boundingSphere_y = (max_y + min_y) / 2
		boundingSphere_z = (max_z + min_z) / 2
		boundingSphere_r = 0

		# increment bounding radius for objects.
		for i in range(len(min_x_)):
			d = self.distance_3d(boundingSphere_x,boundingSphere_y,boundingSphere_z , min_x_[i],min_y_[i],min_z_[i])
			boundingSphere_r = max( boundingSphere_r , d )
		
		aspectX = parent.Viewport.width / parent.Viewport.height
		fovX = CameraCOMP.par.fov.eval()
		fov_min = min( fovX , fovX/aspectX )
		fov_min_half = fov_min / 2
		fov_tan = math.tan( math.radians(fov_min_half) )

		dist_from_object = boundingSphere_r / fov_tan

		BoundingSphere = tdu.Position(boundingSphere_x,boundingSphere_y,boundingSphere_z)
		camera_position_offset = BoundingSphere - (ViewDirection * dist_from_object)

		ViewMatrix[0,3] = camera_position_offset[0]
		ViewMatrix[1,3] = camera_position_offset[1]
		ViewMatrix[2,3] = camera_position_offset[2]

		CameraCOMP.CamMatrix = ViewMatrix
		CameraCOMP.Pivot = tdu.Vector(BoundingSphere)
		CameraCOMP.Update_Camera()

		return