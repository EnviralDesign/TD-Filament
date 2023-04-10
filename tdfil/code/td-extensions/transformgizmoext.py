import numpy as np
import math
import TDFunctions as TDF

class transformgizmoext:
	"""
	gizmoext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.SourcePanelCOMP = self.ownerComp.par.Editorcomp.eval()
		self.SceneCOMP = self.ownerComp.par.Scenecomp.eval()

		self.ownerComp.par.Visible = self.ownerComp.par.Visible.default # true
		self.ownerComp.par.Canceled = self.ownerComp.par.Canceled.default # false

		# used when neccesary for rotation around center sphere, trackball style.
		self.ArcBall = tdu.ArcBall(forCamera=False)

		# properties
		TDF.createProperty(self, 'picking_active', value=False, dependable=True,readOnly=False)
		TDF.createProperty(self, 'element_id', value=False, dependable=True,readOnly=False)

		# initialize some camera matricies. 
		CameraCOMP = self.ownerComp.par.Cameracomp.eval()
		if CameraCOMP == None:
			debug('Camera COMP is invalid, please supply a valid camera to the gizmo.')
		self.camera_position = tdu.Position(CameraCOMP.worldTransform.decompose()[2])
		self.camera_projection_matrix = tdu.Matrix()
		self.camera_projection_inverse_matrix = tdu.Matrix()
		self.camera_view_matrix = tdu.Matrix()
		self.camera_view_inverse_matrix = tdu.Matrix()

		# chops containing the gizmo's normalized direction vectors.
		self.vx_chop = self.ownerComp.op('vx')
		self.vy_chop = self.ownerComp.op('vy')
		self.vz_chop = self.ownerComp.op('vz')

		self.orientation_trigger_chop = self.ownerComp.op('trigger_orientation_change')

		# any objects the gizmo is responsible for affecting should be in this list.
		self.affected_objects = []

		# when the gizmo is initialized, this dictionary should be updated with relevant initial parameters like translate, rotate, scale.
		self.initial_object_params = {}

		# the gizmo's location. we could just query the tx,ty,tz params, but this gives us a static record we can refer to of where the gizmo was at initialization.
		self.initial_gizmo_location = tdu.Position(self.ownerComp.par.tx,self.ownerComp.par.ty,self.ownerComp.par.tz)

		# this gets updated when the Begin() command is called, it stores the offset of the gizmo's interaction, so that we can displace FROM this, relatively.
		self.initial_gizmo_drag = tdu.Position(0)

		# the element index of the initial gizmo handle clicked on.
		self.initial_gizmo_element = 0

		# the gizmo vectors are for world space transforms always going to look like the below initialization values.
		# however, if the user changes the gizmo mode to local, these will look different.
		self.gizmo_vector_x = tdu.Vector(1,0,0)
		self.gizmo_vector_y = tdu.Vector(0,1,0)
		self.gizmo_vector_z = tdu.Vector(0,0,1)

		# this vector points to the camera from the gizmo origin always.
		# NOTE, while this is formatted correctly, it produces incorrect results because camera_view_inverse_matrix is just an identity matrix at this point.
		# this is fine, since we will for sure update this again before the user actually gets to move anything.
		self.gizmo_vector_w = self.camera_position - ( self.camera_view_matrix * tdu.Position(0,0,-1) )
		self.gizmo_vector_w.normalize()

		# this is meant to be updated every time a transform is initialized, but not all transform modes make use of it.
		# it is intended to point from the gizmo origin to the mouse in 2d uv space.
		self.gizmo_vector2d_giz2mouse = tdu.Vector(1,0,0)

		# the active axis and action are updated whenever the Begin() function is called, but we initialize them to some sensible defaults here.
		self.gizmo_active_axis = self.gizmo_vector_y
		self.gizmo_inverse_axis = False # when user drags the xy plane gizmo handle for ex, this will be True.
		self.gizmo_active_action = 'translate'

		self.gizmo_orientation_mode = self.ownerComp.par.Transformationorientation.eval()
		self.orientation_trigger_chop.par.triggerpulse.pulse()


	############################
	### PRIVATE FUNCTIONS
	############################

	def uv_2_pixel(self, pos):
		'''
		returns pixel scaled coordinates from uv coordinates.
		'''
		pos = pos.copy()
		pos.x = pos.x * self.SourcePanelCOMP.width
		pos.y = pos.y * self.SourcePanelCOMP.height
		return pos

	def uv_2_ndc(self, pos, optional_ndc_z=None):
		'''
		converts normalized uv coordinates into ndc space coordinates.
		NOTE: while this function expects pos to be a tdu.Position() object
		only the x 
		'''
		pos = pos.copy()
		pos.x = pos.x * 2 - 1
		pos.y = pos.y * 2 - 1
		if optional_ndc_z != None:
			pos.z  = optional_ndc_z
		else:
			pos.z = 0.98 # 0.98 is arbitrary.
		return pos

	def ndc_2_uv(self, pos):
		'''
		returns uv position from ndc coord.
		'''
		pos = pos.copy()
		pos.x = pos.x * .5 + .5
		pos.y = pos.y * .5 + .5
		pos.z = 0
		return pos

	def ndc_2_view(self, pos):
		'''
		converts a coordinate from ndc space to view space.
		this assumes we are using the matricies that are provided via the camera parameter COMP.
		'''
		pos = pos.copy()
		return self.camera_projection_inverse_matrix * pos

	def view_2_ndc(self, pos):
		'''
		converts a coordinate from view space to ndc space.
		this assumes we are using the matricies that are provided via the camera parameter COMP.
		'''
		pos = pos.copy()
		return self.camera_projection_matrix * pos

	def view_2_world(self, pos):
		'''
		converts a coordinate from view space to world space.
		this assumes we are using the matricies that are provided via the camera parameter COMP.
		'''
		pos = pos.copy()
		return self.camera_view_matrix * pos

	def world_2_view(self, pos):
		'''
		converts a coordinate from world space to view space.
		this assumes we are using the matricies that are provided via the camera parameter COMP.
		'''
		pos = pos.copy()
		return self.camera_view_inverse_matrix * pos
	
	def world_2_uv(self, pos):
		pos = pos.copy()
		return self.ndc_2_uv(self.view_2_ndc(self.world_2_view(pos)))
	
	def uv_2_world(self, pos):
		pos = pos.copy()
		return self.view_2_world(self.ndc_2_view(self.uv_2_ndc(pos)))

	def calculate_angle_between_vectors_360(self, v1, v2):
		'''
		returns the difference between the angles as 360 degrees not 180, input vectors need not be normalized.
		input vectors should be in format tdu.Vector(), though we're not using the classes methods here.
		'''

		v1.normalize()
		v2.normalize()

		x1,y1 = v1.x,v1.y
		x2,y2 = v2.x,v2.y

		angle = math.atan2(y1, x1) - math.atan2(y2, x2)
		angle = angle * 360 / (2.0*math.pi)

		if(angle < 0):
			angle += 360

		return angle
	
	def calculate_ray_plane_intersect_3D(self, planePoint , planeNormal , rayPoint , rayDirection ):
		'''
		planePoint = any point on the imaginary plane that we want to raycast to. easiest to think of this as the "center" of the plane.
		planeNormal = the normal direction of the plane. this is a more useful way of describing which way the plane is facing, as if we rotated it.
		rayPoint = the starting point of our ray. for the purposes of this application, this will always be the camera's origin / position in world space.
		rayDirection = direction of the ray, meaning outside of this function, we should calculate this argument like this: ( pointOnNearClipInWs - camOriginInWs )
		'''
		
		# used to detect if the ray is perpindicular to the plane it's supposed to intersect.
		epsilon=1e-6
		
		# vectorize the plane Normal, and normalize it for length of 1.
		planeNormal = tdu.Vector( planeNormal )
		planeNormal.normalize()
		
		# turn the plane point into a tdu Position.
		planePoint = tdu.Position( planePoint )
		
		# vectorize the ray direction, and normalize it for length of 1.
		rayDirection = tdu.Vector( rayDirection )
		rayDirection.normalize()
		
		# any point along the ray will do, but the most intuitive value to use here is the camera origin / position in WS.
		rayPoint = tdu.Position( rayPoint )
		
		# dot product of plane normal and ray direction.
		ndotu = planeNormal.dot(rayDirection) 
		
		# handle edge case where there is no possible intersection due to ray being parallel with plane.
		if abs(ndotu) < epsilon:
			debug("no intersection or line is within plane")
		
		# finish calculating the point of intersection for the ray and plane, resulting in a tdu.Position object.
		w = rayPoint - planePoint
		si = -planeNormal.dot(w) / ndotu
		Psi = w + si * rayDirection + planePoint
		
		return Psi

	def calculate_closest_point_on_line_2D(self, a0, a1, b):
		'''
		a0 and a1 define a line, and b defines a single point. this function returns a new point that
		lies on the line defined by a0/a1, that is closest to b.
		NOTE: this function deals in 2d coordinates only, so the .z member of the tdu.Position() object is discarded.
		https://blender.stackexchange.com/questions/94464/finding-the-closest-point-on-a-line-defined-by-two-points
		'''

		# make sure we dont have 3d data on z.
		a0.z = 0
		a1.z = 0
		b.z = 0

		n = (a1 - a0)
		n.normalize()

		ab = b - a0
		t = ab.dot(n)

		x =  a0 + t * n

		return x

	def calculate_shortest_line_between_lines_3D(self, a0,a1,b0,b1,clampAll=False,clampA0=False,clampA1=False,clampB0=False,clampB1=False):

		''' Given two lines defined by two tdu.Position() objects,
			Return the closest points on each segment and their distance
		'''

		# converte results to numpy arrays.
		a0 = np.array( [ a0.x , a0.y , a0.z ] )
		a1 = np.array( [ a1.x , a1.y , a1.z ] )
		b0 = np.array( [ b0.x , b0.y , b0.z ] )
		b1 = np.array( [ b1.x , b1.y , b1.z ] )

		# If clampAll=True, set all clamps to True
		if clampAll:
			clampA0=True
			clampA1=True
			clampB0=True
			clampB1=True

		# Calculate denomitator
		A = a1 - a0
		B = b1 - b0
		magA = np.linalg.norm(A)
		magB = np.linalg.norm(B)
		
		_A = A / magA
		_B = B / magB
		
		cross = np.cross(_A, _B)
		denom = np.linalg.norm(cross)**2
		
		# If lines are parallel (denom=0) test if lines overlap.
		# If they don't overlap then there is a closest point solution.
		# If they do overlap, there are infinite closest positions, but there is a closest distance
		if not denom:
			d0 = np.dot(_A,(b0-a0))
			
			# Overlap only possible with clamping
			if clampA0 or clampA1 or clampB0 or clampB1:
				d1 = np.dot(_A,(b1-a0))
				
				# Is segment B before A?
				if d0 <= 0 >= d1:
					if clampA0 and clampB1:
						if np.absolute(d0) < np.absolute(d1):
							return a0,b0,np.linalg.norm(a0-b0)
						return a0,b1,np.linalg.norm(a0-b1)
					
				# Is segment B after A?
				elif d0 >= magA <= d1:
					if clampA1 and clampB0:
						if np.absolute(d0) < np.absolute(d1):
							return a1,b0,np.linalg.norm(a1-b0)
						return a1,b1,np.linalg.norm(a1-b1)
					
			# Segments overlap, return distance between parallel segments
			return None,None,np.linalg.norm(((d0*_A)+a0)-b0)
		
		# Lines criss-cross: Calculate the projected closest points
		t = (b0 - a0);
		detA = np.linalg.det([t, _B, cross])
		detB = np.linalg.det([t, _A, cross])

		t0 = detA/denom;
		t1 = detB/denom;

		pA = a0 + (_A * t0) # Projected closest point on segment A
		pB = b0 + (_B * t1) # Projected closest point on segment B


		# Clamp projections
		if clampA0 or clampA1 or clampB0 or clampB1:
			if clampA0 and t0 < 0:
				pA = a0
			elif clampA1 and t0 > magA:
				pA = a1
			
			if clampB0 and t1 < 0:
				pB = b0
			elif clampB1 and t1 > magB:
				pB = b1
				
			# Clamp projection A
			if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
				dot = np.dot(_B,(pA-b0))
				if clampB0 and dot < 0:
					dot = 0
				elif clampB1 and dot > magB:
					dot = magB
				pB = b0 + (_B * dot)
		
			# Clamp projection B
			if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
				dot = np.dot(_A,(pB-a0))
				if clampA0 and dot < 0:
					dot = 0
				elif clampA1 and dot > magA:
					dot = magA
				pA = a0 + (_A * dot)

		
		return tdu.Position(pA) , tdu.Position(pB) , np.linalg.norm(pA-pB)

	def calculate_percentage_along_3D_line(self, a0, a1, uv):
		'''
		a0 = a 3d point (tdu.Position) representing the location of the gizmo.
		a1 = a 3d point (tdu.Position) derived from the location of the gizmo, displaced in the direction of the active axis vector.
		uv = a 2d point (tdu.Position, .z discarded) representing the 0:1 normalized position of the cursor on screen.
		'''

		# convert our gizmo axis line which is comprised of a0 and a1 from world space to 0-1 uv space.
		a0_ndc = self.view_2_ndc( self.world_2_view(a0) )
		a1_ndc = self.view_2_ndc( self.world_2_view(a1) )
		a0_uv = self.ndc_2_uv( a0_ndc )
		a1_uv = self.ndc_2_uv( a1_ndc )
		#TODO: maybe use the aspect ratio to scale uv to get a more accurate closest point???

		# self.DEBUG_DRAW_LINE_2D(0,a0_uv,a1_uv)

		# find the closest point on our 2d line, to our mouse uv coordinate.
		result = self.calculate_closest_point_on_line_2D( a0_uv , a1_uv , uv )
		
		# self.DEBUG_DRAW_LINE_2D(1,uv,result)
		# self.DEBUG_DRAW_SPHERE_2D(0,uv)

		# get length of the vector pointing from the gizmo to the gizmo vector in uv space.
		segment_vector = a1_uv-a0_uv
		segment_length = segment_vector.length()

		# get length of the vector pointing from the gizmo to the gizmo closest point on line.
		position_vector = result-a0_uv
		position_length = position_vector.length()
		
		# now that we got the lengths of these vectors, we can normalize them.
		segment_vector.normalize()
		position_vector.normalize()

		# calc the dot product between these vectors. since both vectors should always be aligned
		# due to the way they are derived, we can assume the dot product will be ~ 1 or ~-1 with some
		# small rounding errors. This gives us the sign, or rather tells us if the user dragged
		# the gizmo in the positive direction of it's axis, or the negative direction.
		# We are cleaning up the signal by doing a boolean test, converting to int, then rescaling to -1:1.
		sign = int(segment_vector.dot(position_vector) >= 0) * 2 - 1

		# now we can finally calcualte our normalized 0-1 percentage along the long defined by our gizmo and gizmo vector.
		# NOTE: this normalized value can and should exceed 0:1 dependng on how far the gizmo is dragged and in what local direction.
		percentage_along_line = ( position_length / segment_length ) * sign

		return percentage_along_line

	def calculate_3D_position_of_closest_point_on_line(self, a0, a1, uv):
		'''
		a0 = a 3d point (tdu.Position) representing the location of the gizmo.
		a1 = a 3d point (tdu.Position) derived from the location of the gizmo, displaced in the direction of the active axis vector.
		uv = a 2d point (tdu.Position, .z discarded) representing the 0:1 normalized position of the cursor on screen.
		'''

		a0_ndc = self.view_2_ndc( self.world_2_view(a0) )
		a1_ndc = self.view_2_ndc( self.world_2_view(a1) )

		# now we can finally calcualte our normalized 0-1 percentage along the long defined by our gizmo and gizmo vector.
		# NOTE: this normalized value can and should exceed 0:1 dependng on how far the gizmo is dragged and in what local direction.
		percentage_along_line = self.calculate_percentage_along_3D_line(a0, a1, uv)

		# now we can lerp between our two ndc world space coordinates using the percentage, and this will
		# give us a correctly interpolated position of how far we've dragged the gizmo along the line.
		# ndc_space_coordinate = tdu.Vector( a0_ndc ).lerp( tdu.Vector( a1_ndc ) , percentage_along_line )
		ndc_space_coordinate = tdu.Position(
			tdu.remap(percentage_along_line , 0 , 1 , a0_ndc.x , a1_ndc.x),
			tdu.remap(percentage_along_line , 0 , 1 , a0_ndc.y , a1_ndc.y),
			tdu.remap(percentage_along_line , 0 , 1 , a0_ndc.z , a1_ndc.z)
		)
		# op('table_debug').clear()
		# op('table_debug').appendRow([a0_ndc.x,a0_ndc.y,a0_ndc.z])
		# op('table_debug').appendRow([a1_ndc.x,a1_ndc.y,a1_ndc.z])
		# op('table_debug').appendRow([ndc_space_coordinate.x,ndc_space_coordinate.y,ndc_space_coordinate.z])
		
		# once we get our ndc space iterpolated coordinate (ndc space is linear), then we can convert it back to world space
		# which takes into account projection depth, which is non linear.
		world_space_coordinate = self.view_2_world( self.ndc_2_view( ndc_space_coordinate ) )

		return tdu.Position( world_space_coordinate )
	
	def calculate_transform_of_current_object(self):
		
		affected_objects = self.affected_objects

		if len(affected_objects) == 0:
			# debug('no affected objects to check, skipping...')
			return
		
		current_object = [ x for x in affected_objects if x.current == True ]
		# print('CURRENT', current_object)
		if len(current_object) == 1:
			Obj = current_object[0]
		
		else:
			# debug('non of the affected objects are marked as "current", this is likely a bug. Using last object in affected_objects list...')
			# print('----------',affected_objects)
			Obj = affected_objects[-1]
			# Obj.current = True
		
		s,r,t = Obj.worldTransform.decompose()


		return {'t':t, 'r':r}

	def calculate_averge_center_of_affected_objects(self):
		'''
		Calculates the average center of the currently attached affected objects. 
		This will use the transform's center as the points that get averaged, not the bound boxes or verticies.

		This uses world space position to calculate averge center.
		'''

		# get gizmos objects.
		affected_objects = self.affected_objects

		# calc average.
		x,y,z = [],[],[]
		for each in self.affected_objects:
			m = each.worldTransform
			s,r,t = m.decompose()
			x += [t[0]]
			y += [t[1]]
			z += [t[2]]
		numObjects = len(self.affected_objects)

		if numObjects > 0:
			x = sum(x)/numObjects
			y = sum(y)/numObjects
			z = sum(z)/numObjects
		else:
			x = 0
			y = 0
			z = 0
		
		# return average center as position
		return tdu.Position(x,y,z)

	def translate_uv_2_xyz( self, u, v ):
		'''
		this is the top level wrapper for translation function that returns an xyz position for the closest point on 
		the active gizmo line / plane.
		'''
		# coordinate of where the gizmo is in 3d world space.
		a0 = self.initial_gizmo_location

		# a second coordinate that makes up a line, the gizmo active axis is a world space vector, 
		# so we add it to the initial position to derive a second position on the line.
		a1 = a0 + self.gizmo_active_axis

		# define the point in uv space we want to test against our 2d gizmo line.
		# this is derived from our mouse position using panel CHOP.
		b_2d = tdu.Position( u , v , 0 )

		# if the user clicked and dragged one of the 3 translate arrows.
		if self.gizmo_inverse_axis == False:
			closest_3d_point = self.calculate_3D_position_of_closest_point_on_line( a0 , a1 , b_2d )

		# if the user clicked and dragged one of the 3 translate blocks for 2 axis constrained movement.
		elif self.gizmo_inverse_axis == True:
			closest_3d_point = self.calculate_ray_plane_intersect_3D( a0 , self.gizmo_active_axis , self.camera_position , self.uv_2_world( b_2d )-self.camera_position )

		return closest_3d_point

	def rotate_uv_2_angle(self, u, v):
		'''
		this is the top level wrapper for rotation function that returns an angle in degrees that the cursor is rotated around 
		the active gizmo point as it exists in screen space. (2d). 
		'''
		
		# define the 2d uv coord as a 3d position even though we're not using the third component here.
		point_ss = tdu.Position( u , v , 0 )

		# coordinate of where the gizmo is in 3d world space.
		pivot_ws = self.initial_gizmo_location
		
		# convert the world space pivot (gizmo origin) to a 0:1 place in screen space.
		pivot_ss = self.world_2_uv(pivot_ws)
		pivot_ss.z = 0 # don't need z, lets ensure it's set to zero.

		# offset our user point to center of 2d space, using the pivot, then normalize.
		v1 = point_ss - pivot_ss; v1.normalize()

		# define our reference vector, as a unit vector pointing to the right in 2d space, along X.
		v2 = tdu.Vector(1,0,0)

		# calc the angle between vectors, in degrees.
		angle_degrees = self.calculate_angle_between_vectors_360( v1 , v2 )

		return angle_degrees

	def rotate_uv_2_arcball(self, u, v):
		'''
		arcball rotation function.
		'''
		rotation_sensitivity = 3.14159 # this number felt good, does not necessarily have anything to do with pi. 
		self.ArcBall.rotateTo(u,v,scale=rotation_sensitivity)
		arcball_mat = self.ArcBall.transform()

		# must rotate the arcball transform by the cameras rotation for it to always rotate
		# in a natural manner regardless of view direction.
		camera_rotation = self.camera_view_matrix.decompose()[1]
		arcball_mat.rotate(camera_rotation)

		return arcball_mat

	def scale_uv_2_magnitude(self, u, v):
		'''
		this is intended to be used when the user is trying to scale using the uniform scale grey circle
		instead of a particular plane or axis scaling option.
		'''

		plane_intersect = self.calculate_ray_plane_intersect_3D( 
			self.initial_gizmo_location , 
			self.gizmo_vector_w , 
			self.camera_position , 
			self.ndc_2_view(tdu.Position(u*2-1, v*2-1, 0.98))
			)

		plane_intersect = tdu.Vector(plane_intersect)
		plane_intersect.normalize()

		# coordinate of where the gizmo is in 3d world space.
		a0 = self.initial_gizmo_location

		# a second coordinate that makes up a line, the gizmo active axis is a world space vector, 
		# so we add it to the initial position to derive a second position on the line.
		a1 = a0 + plane_intersect

		# define the point in uv space we want to test against our 2d gizmo line.
		# this is derived from our mouse position using panel CHOP.
		b_2d = tdu.Position( u , v , 0 )

		magnitude = self.calculate_percentage_along_3D_line( a0 , a1 , b_2d )

		return magnitude

	def scale_uv_2_xyz( self, u, v ):
		'''
		this is the top level wrapper for scaling function that returns an xyz position for the closest point on 
		the active gizmo line / plane.
		'''
		# coordinate of where the gizmo is in 3d world space.
		a0 = self.initial_gizmo_location

		# a second coordinate that makes up a line, the gizmo active axis is a world space vector, 
		# so we add it to the initial position to derive a second position on the line.
		a1 = a0 + self.gizmo_active_axis

		# define the point in uv space we want to test against our 2d gizmo line.
		# this is derived from our mouse position using panel CHOP.
		b_2d = tdu.Position( u , v , 0 )

		# if the user clicked and dragged one of the 3 translate arrows.
		if self.gizmo_inverse_axis == False:
			closest_3d_point = self.calculate_3D_position_of_closest_point_on_line( a0 , a1 , b_2d )

		# if the user clicked and dragged one of the 3 translate blocks for 2 axis constrained movement.
		elif self.gizmo_inverse_axis == True:
			closest_3d_point = self.calculate_ray_plane_intersect_3D( a0 , self.gizmo_active_axis , self.camera_position , self.uv_2_world( b_2d )-self.camera_position )

		return closest_3d_point

	def update_gizmo_mouse_vector(self, u, v):
		a = self.world_2_uv( self.initial_gizmo_location )
		b = tdu.Position(u, v, 0)
		self.gizmo_vector2d_giz2mouse = b - a
		self.gizmo_vector2d_giz2mouse.normalize()
		return

	def update_gizmo_position(self):
		'''
		given a tdu.Position object, this function will update the gizmo's position in 3d space 
		and also update the initial gizmo location attribute.
		'''

		if self.gizmo_orientation_mode == 'world':

			avgCenter = self.calculate_averge_center_of_affected_objects()

			self.ownerComp.par.tx = avgCenter.x
			self.ownerComp.par.ty = avgCenter.y
			self.ownerComp.par.tz = avgCenter.z

			self.ownerComp.par.rx = 0
			self.ownerComp.par.ry = 0
			self.ownerComp.par.rz = 0
		
		if self.gizmo_orientation_mode == 'local':
			current_xform = self.calculate_transform_of_current_object()

			if current_xform != None:

				self.ownerComp.par.tx = current_xform['t'][0]
				self.ownerComp.par.ty = current_xform['t'][1]
				self.ownerComp.par.tz = current_xform['t'][2]

				self.ownerComp.par.rx = current_xform['r'][0]
				self.ownerComp.par.ry = current_xform['r'][1]
				self.ownerComp.par.rz = current_xform['r'][2]
			
			else:

				self.ownerComp.par.tx = 0
				self.ownerComp.par.ty = 0
				self.ownerComp.par.tz = 0

				self.ownerComp.par.rx = 0
				self.ownerComp.par.ry = 0
				self.ownerComp.par.rz = 0
		
		self.initial_gizmo_location = tdu.Position( self.ownerComp.par.tx , self.ownerComp.par.ty , self.ownerComp.par.tz )
				

		return
	
	def update_gizmo_vectors(self):
		'''
		depending on if we are using the gizmo in world space where it is always aligned to global axis, 
		or if we are using it in local space where it takes the orientation of the last selected object, 
		we call this function to update the gizmo's internal axis vectors from the chops.
		'''
		self.vx_chop.cook(force=True)
		self.vy_chop.cook(force=True)
		self.vz_chop.cook(force=True)

		self.gizmo_vector_x = tdu.Vector( self.vx_chop[0] , self.vx_chop[1] , self.vx_chop[2] )
		self.gizmo_vector_y = tdu.Vector( self.vy_chop[0] , self.vy_chop[1] , self.vy_chop[2] )
		self.gizmo_vector_z = tdu.Vector( self.vz_chop[0] , self.vz_chop[1] , self.vz_chop[2] )

		# this vector is essentially a ray from the camera to the center of near plane, 
		# however it is reversed and facing the camera.
		self.gizmo_vector_w = self.camera_position - ( self.camera_view_matrix * tdu.Position(0,0,-1) )
		self.gizmo_vector_w.normalize()

		return

	def update_initial_object_params(self):
		'''
		stores a bunch of object attributes to a dictionary structure 
		that we can use as a transform basis during drag operations.
		'''
		fullDict = {}
		for each in self.affected_objects:
			
			# we need to know if we're dealing with an instance or object, because we fetch their parent objects
			# differently. objects we get via wire parent, and instances via container parent 2 levels up.
			element_type = self.SceneCOMP.Element_Type(each)
			
			if element_type == 'obj':

				if len(each.inputCOMPs): # if there's a parent object, get it's world transform too.
					m = each.localTransform
					parent_matrix = each.inputCOMPs[0].worldTransform
				else: # if no parent object, just use an identity matrix so no effect is made.
					m = each.worldTransform
					parent_matrix = tdu.Matrix()
			
			elif element_type == 'inst':
				m = each.localTransform
				parent_matrix = each.parent.obj.worldTransform
			
			d = {}
			d['object_matrix'] = m
			d['parent_matrix'] = parent_matrix.copy(); parent_matrix.invert()
			d['parent_matrix_inverse'] = parent_matrix

			d['tx'] = each.par.tx.eval()
			d['ty'] = each.par.ty.eval()
			d['tz'] = each.par.tz.eval()

			d['rx'] = each.par.rx.eval()
			d['ry'] = each.par.ry.eval()
			d['rz'] = each.par.rz.eval()

			d['sx'] = each.par.sx.eval()
			d['sy'] = each.par.sy.eval()
			d['sz'] = each.par.sz.eval()

			# object vectors just tell us which way the object's pos X, pos Y, and pos Z axis are facing in world space.
			d['vx'] = each.worldTransform * tdu.Vector(1,0,0)
			d['vy'] = each.worldTransform * tdu.Vector(0,1,0)
			d['vz'] = each.worldTransform * tdu.Vector(0,0,1)

			fullDict[each.id] = d
		
		self.initial_object_params = fullDict

		return

	def update_active_axis(self, axisID):
		'''
		depending on which axis the user clicked, this function is eventually called to store the axis the user is trying to interact with.
		'''

		self.initial_gizmo_element = self.element_id

		if axisID in [0,3]:
			self.gizmo_active_axis = self.gizmo_vector_x.copy()
		elif axisID in [1,4]:
			self.gizmo_active_axis = self.gizmo_vector_y.copy()
		elif axisID in [2,5]:
			self.gizmo_active_axis = self.gizmo_vector_z.copy()
		elif axisID in [6]:
			self.gizmo_active_axis = self.gizmo_vector_w.copy()
		elif axisID in [7]:
			self.gizmo_active_axis = None # trackball mode doesn't use a rotation angle determined by vector.
		
		if axisID in [3,4,5,6,7]:
			self.gizmo_inverse_axis = True
		elif axisID in [0,1,2]:
			self.gizmo_inverse_axis = False
		return

	def update_active_action(self, actionID):
		'''
		depending on which gizmo part the user clicked, this function is called to store the action the user is trying to perform.
		'''
		if actionID == 0:
			self.gizmo_active_action = 'translate'
		
		elif actionID == 1:
			self.gizmo_active_action = 'rotate'
		
		elif actionID == 2:
			self.gizmo_active_action = 'scale'
		return

	def update_camera_data(self):
		'''
		update the gizmo with information about the camera, namely view and projection matricies as well as their inverses.
		'''
		CameraCOMP = self.ownerComp.par.Cameracomp.eval()
		if CameraCOMP == None:
			debug('camera invalid, cannot proceed...')
			return

		cameraViewMatrix = CameraCOMP.worldTransform
		cameraViewInvMatrix = cameraViewMatrix.copy()
		cameraViewInvMatrix.invert()

		cameraProjInvMatrix = CameraCOMP.projectionInverse( self.SourcePanelCOMP.width , self.SourcePanelCOMP.height )
		cameraProjMatrix = cameraProjInvMatrix.copy()
		cameraProjMatrix.invert()

		s,r,t = cameraViewMatrix.decompose()

		self.camera_position = tdu.Position( t )
		self.camera_projection_matrix = cameraProjMatrix.copy()
		self.camera_projection_inverse_matrix = cameraProjInvMatrix.copy()
		self.camera_view_matrix = cameraViewMatrix.copy()
		self.camera_view_inverse_matrix = cameraViewInvMatrix.copy()

		return
	
	############################
	### PUBLIC FUNCTIONS
	############################
	def Update_Picking_Status(self, status):
		self.picking_active = status
		return

	@property
	def Picking_Active(self):
		# return self.picking_active
		return self.picking_active * self.ownerComp.par.Visible

	def Update_Picking_Element(self, element_id):
		self.element_id = element_id
		return

	@property
	def Picking_Element(self):
		return self.element_id
	
	def Update_Transform_Orientation(self):
		self.gizmo_orientation_mode = self.ownerComp.par.Transformationorientation.eval()
		self.Init(self.affected_objects)
		self.orientation_trigger_chop.par.triggerpulse.pulse()
		return

	def Set_Mode(self, modeString):
		'''	
		modeString = ['select','translate','rotate','scale']
		'''
		currentModeString = self.ownerComp.par.Transformationmode.eval()
		newModeString = modeString.lower()
		self.ownerComp.par.Transformationmode = newModeString

		if newModeString == currentModeString:
			self.ownerComp.par.Transformationorientation.menuIndex = 1 - self.ownerComp.par.Transformationorientation.menuIndex

		else:
			# if we are changing to a dif mode still pulse the orientation mode to remind the user what they are in.
			self.orientation_trigger_chop.par.triggerpulse.pulse()

		return
	
	def Init(self, targetObjects=None):
		'''
		Init() should be called whenever the selection of objects is changed, the gizmo is enabled, etc.
		This function preloads the gizmo with the provided target objects, and also resets/updates some parameters.
		'''

		# store the orientation mode every time we initialize the gizmo.
		self.gizmo_orientation_mode = self.ownerComp.par.Transformationorientation.eval()

		# reset to defaults.
		self.ownerComp.par.Visible = self.ownerComp.par.Visible.default # true
		self.ownerComp.par.Canceled = self.ownerComp.par.Canceled.default # false

		# update affected objects list only if a new list is provided.
		if targetObjects != None:
			self.affected_objects = targetObjects
		
			if len(targetObjects) == 0:
				self.ownerComp.par.Visible = False
		
		# calculate the average center of all the objects.
		#TODO: update_gizmo_position() could take an argument to determine if it's using average center, last selected xform, etc..
		self.update_gizmo_position()

		return

	def Begin(self, axisID, actionID , u , v ):
		'''
		Begin() is meant to be called from the render pick callback or similar, when a down click begins, when user is trying to move gizmo. 
		It's designed to cache a lot of heavy lifting up front, so the Drag function can operate in a more streamlined fashion.
		'''
		self.ownerComp.par.Visible = False

		# update gizmo with latest camera data.
		self.update_camera_data()

		# update gizmo's axis vectors. NOTE: must be called after update_camera_data()
		self.update_gizmo_vectors()

		# update the axis id with what was clicked on.
		self.update_active_axis(axisID)

		# update the action id with what was clicked on.
		self.update_active_action(actionID)

		# update the gizmo to mouse 2d uv vector.
		self.update_gizmo_mouse_vector( u , v )

		if self.gizmo_active_action == 'translate':
			# set an initial drag position for our objects.
			self.initial_gizmo_drag = self.translate_uv_2_xyz( u , v )
		
		elif self.gizmo_active_action == 'rotate':
			# AXIS ROTATION
			if isinstance(self.gizmo_active_axis, tdu.Vector):
				# set an initial rotation angle for our objects.
				self.initial_gizmo_drag = self.rotate_uv_2_angle( u , v )
			
			# TRACKBALL ROTATION
			else:
				# set an initial rotation angle for our objects.
				self.ArcBall.identity()
				self.ArcBall.beginRotate(u,v)
				self.initial_gizmo_drag = self.rotate_uv_2_arcball( u , v )
		
		elif self.gizmo_active_action == 'scale':
			
			# set an initial drag position for our objects.
			if self.initial_gizmo_element != 7:
				self.initial_gizmo_drag = self.scale_uv_2_xyz( u , v )
			else:
				self.initial_gizmo_drag = self.scale_uv_2_magnitude( u , v )

		# update gizmo's affected object's initial parameters, translate,rotate,scale, etc.
		self.update_initial_object_params()

		return

	def Drag(self, u , v ):
		'''
		This is called from the callback or similar when the user is dragging their mouse with left click held down.
		'''

		# Canceled is set to true in the event that the user right clicks during a drag. since the render pick still thinks it's
		# engaged, we simply check for this parameter every time drag is called ,and silently bail early. after they release the mouse
		# and try again, Canceled will be set back to default, and all is normal again.
		if self.ownerComp.par.Canceled == True:
			return

		# gizmo should not be visible during xforms. if the visible flag is on, turn it off.
		if self.ownerComp.par.Visible == True:
			self.ownerComp.par.Visible = False

		# logic branch for handling translation.
		if self.gizmo_active_action == 'translate':

			if self.gizmo_orientation_mode == 'world':
			
				# we get the absolute point returned by our ray cast function.
				closest_3d_point = self.translate_uv_2_xyz( u , v )

				# we subtract it from what was stored in the Begin() function. this gives us relative drag, and not absolute.
				displaced_position = closest_3d_point - self.initial_gizmo_drag

				# for each of the selected objects.
				for obj in self.affected_objects:

					# get the initial object dictionary for this object.
					objInitialParams = self.initial_object_params[obj.id]

					# get the inverse matrix of the objects parent, if it has no parent, this will be an identity matrix.
					parent_matrix = objInitialParams['parent_matrix'].copy()
					parent_matrix_inverse = objInitialParams['parent_matrix_inverse'].copy()
					object_matrix = objInitialParams['object_matrix'].copy()

					# first we must take any objects parented to other objects, and transform their matrix into world space.
					# the object_matrix will represent that object in the same transformation as before, just without a parent.
					object_matrix = parent_matrix * object_matrix
					
					# in world space, now we can do our translation by the desired amount.
					object_matrix.translate( displaced_position )

					# now we've done our translation, we need to move our world space matrix back to the space it was in before.
					object_matrix = parent_matrix_inverse * object_matrix

					# we can just set the transform with this function.
					obj.setTransform(object_matrix)

			elif self.gizmo_orientation_mode == 'local':
				
				# we get the absolute point returned by our ray cast function.
				closest_3d_point = self.translate_uv_2_xyz( u , v )

				# we subtract it from what was stored in the Begin() function. this gives us relative drag, and not absolute.
				displaced_position = closest_3d_point - self.initial_gizmo_drag

				gizmo_matrix = self.ownerComp.worldTransform; gizmo_matrix.invert()
				displaced_position_gizmospace = gizmo_matrix * displaced_position

				x_mask = self.gizmo_active_axis.dot(self.gizmo_vector_x); x_mask = 0 if x_mask <= 0.0001 else x_mask
				y_mask = self.gizmo_active_axis.dot(self.gizmo_vector_y); y_mask = 0 if y_mask <= 0.0001 else y_mask
				z_mask = self.gizmo_active_axis.dot(self.gizmo_vector_z); z_mask = 0 if z_mask <= 0.0001 else z_mask

				# if user dragged a plane, invert our axis masks.
				if self.gizmo_inverse_axis == True:
					x_mask = 1 - x_mask
					y_mask = 1 - y_mask
					z_mask = 1 - z_mask

				displaced_position_gizmospace.x *= x_mask
				displaced_position_gizmospace.y *= y_mask
				displaced_position_gizmospace.z *= z_mask

				# for each of the selected objects.
				for obj in self.affected_objects:

					# get the initial object dictionary for this object.
					objInitialParams = self.initial_object_params[obj.id]

					# get the inverse matrix of the objects parent, if it has no parent, this will be an identity matrix.
					parent_matrix = objInitialParams['parent_matrix'].copy()
					parent_matrix_inverse = objInitialParams['parent_matrix_inverse'].copy()
					object_matrix = objInitialParams['object_matrix'].copy()
					object_matrix_rotation_only = tdu.Matrix()
					object_matrix_rotation_only.rotate(object_matrix.decompose()[1])

					# get the initial translate values on the object.
					tx,ty,tz = objInitialParams['tx'],objInitialParams['ty'],objInitialParams['tz']
					rx,ry,rz = objInitialParams['rx'],objInitialParams['ry'],objInitialParams['rz']

					# in object space, now we can do our translation by the desired amount.
					#NOTE: we must multiply our gizmo space xforms by the object matrix,
					# to generate xy/z offsets in world space, since objects in TD apply translation last.
					object_local_displace = displaced_position_gizmospace
					object_local_displace = object_matrix_rotation_only * object_local_displace

					obj.par.tx = tx + object_local_displace.x
					obj.par.ty = ty + object_local_displace.y
					obj.par.tz = tz + object_local_displace.z
		
		# logic branch for handling rotation.
		elif self.gizmo_active_action == 'rotate':

			# AXIS ROTATION
			if isinstance(self.gizmo_active_axis, tdu.Vector):

				# we get the absolute angle returned by our rotation calculation function.
				angle = self.rotate_uv_2_angle( u , v )

				# we subtract it from what was stored in the Begin() function. this gives us relative angle, and not absolute.
				angle = angle - self.initial_gizmo_drag

				if self.gizmo_orientation_mode == 'local':
					# get the gizmo's transform matrix then invert it.
					gizmo_matrix = self.ownerComp.worldTransform; gizmo_matrix.invert()

					# next take our world sapce gizmo active axis, and transform it to local space.
					# this gives us the axis x, y, or z that the user is trying to rotate around.
					gizmo_active_axis_gizmospace = gizmo_matrix * self.gizmo_active_axis
					gizmo_active_axis_gizmospace.normalize()

					# remove really small fluctuations of rotation on other axes, and set them to 0.
					gizmo_active_axis_gizmospace.x = 0 if gizmo_active_axis_gizmospace.x <= 0.001 else gizmo_active_axis_gizmospace.x
					gizmo_active_axis_gizmospace.y = 0 if gizmo_active_axis_gizmospace.y <= 0.001 else gizmo_active_axis_gizmospace.y
					gizmo_active_axis_gizmospace.z = 0 if gizmo_active_axis_gizmospace.z <= 0.001 else gizmo_active_axis_gizmospace.z
				else:
					gizmo_active_axis_gizmospace = tdu.Vector(0)

				# for each of the selected objects.
				for obj in self.affected_objects:

					# get the initial object dictionary for this object.
					objInitialParams = self.initial_object_params[obj.id]

					# get the inverse matrix of the objects parent, if it has no parent, this will be an identity matrix.
					parent_matrix = objInitialParams['parent_matrix'].copy()
					parent_matrix_inverse = objInitialParams['parent_matrix_inverse'].copy()
					object_matrix = objInitialParams['object_matrix'].copy()

					# first we must take any objects parented to other objects, and transform their matrix into world space.
					# the object_matrix will represent that object in the same transformation as before, just without a parent.
					object_matrix = parent_matrix * object_matrix

					if self.gizmo_orientation_mode == 'world':
						
						# in world space, now we can do our rotation around the desired pivot, and around the desired axis, by the desired amount.
						gizmo_active_axis = self.gizmo_active_axis.copy()

						# based on which direction we're rotating around the axis in question from, we may need to flip the rotation direction.
						# we can do this by flipping the signs of the active axis which results in correct rotation from any angle.
						sign = int(gizmo_active_axis.dot(self.gizmo_vector_w) >= 0) * 2 - 1
						gizmo_active_axis.x *= sign
						gizmo_active_axis.y *= sign
						gizmo_active_axis.z *= sign
						
						# gizmo_active_axis.x = -gizmo_active_axis.x # unclear why this is neccesary, but the X rotation axis comes in inverted.
						object_matrix.rotateOnAxis(gizmo_active_axis, angle, fromRight=False, pivot=self.initial_gizmo_location)

						# now we've done our rotation, we need to move our world space matrix back to the space it was in before.
						object_matrix = parent_matrix_inverse * object_matrix
						
						# we can just set the transform with this function.
						obj.setTransform(object_matrix)
			
					
					elif self.gizmo_orientation_mode == 'local':
						
						# get the world space position of the object, this will be our rotation pivot.
						t = object_matrix.decompose()[2]
						
						rotation_vector = object_matrix * gizmo_active_axis_gizmospace

						# based on which direction we're rotating around the axis in question from, we may need to flip the rotation direction.
						# we can do this by flipping the signs of the active axis which results in correct rotation from any angle.
						sign = int(rotation_vector.dot(self.gizmo_vector_w) >= 0) * 2 - 1
						rotation_vector.x *= sign
						rotation_vector.y *= sign
						rotation_vector.z *= sign

						# in world space, now we can do our rotation around the desired pivot, and around the desired axis, by the desired amount.
						object_matrix.rotateOnAxis(rotation_vector, angle, fromRight=False, pivot=t)

						# we can just set the transform with this function.
						obj.setTransform(object_matrix)

			# TRACKBALL ROTATION
			else:
				
				# get current arcball rotation, and mult it against inverse of initial arcball matrix
				# to neutralize the offset, and get a relative matrix.
				arcball_matrix = self.rotate_uv_2_arcball( u , v )
				initarcball_matrix_inverse = self.initial_gizmo_drag.copy()
				initarcball_matrix_inverse.invert()
				arcball_relative_offset_matrix = arcball_matrix * initarcball_matrix_inverse

				# get just the rotation values of the arcball matrix.
				arcball_rotation_values = arcball_relative_offset_matrix.decompose()[1]
				
				# for each of the selected objects.
				for obj in self.affected_objects:

					# get the initial object dictionary for this object.
					objInitialParams = self.initial_object_params[obj.id]

					# get the inverse matrix of the objects parent, if it has no parent, this will be an identity matrix.
					parent_matrix = objInitialParams['parent_matrix'].copy()
					parent_matrix_inverse = objInitialParams['parent_matrix_inverse'].copy()
					object_matrix = objInitialParams['object_matrix'].copy()

					if self.gizmo_orientation_mode == 'world':

						# first we must take any objects parented to other objects, and transform their matrix into world space.
						# the object_matrix will represent that object in the same transformation as before, just without a parent.
						object_matrix = parent_matrix * object_matrix

						# now we must offset our matrix so that the pivot is manually centered at gizmo origin.
						object_matrix.translate( -self.initial_gizmo_location )
						
						# in gizmo space, now we can rotate our object's matrix by the arcball rotation offset amounts.
						object_matrix.rotate( arcball_rotation_values )

						# now we must un-offset our matrix.
						object_matrix.translate( self.initial_gizmo_location )

						# now we've done our rotation, we need to move our world space matrix back to the space it was in before.
						object_matrix = parent_matrix_inverse * object_matrix
					
					elif self.gizmo_orientation_mode == 'local':

						# get the world space position of the object, this will be our rotation pivot.
						t = tdu.Position( object_matrix.decompose()[2] )

						# now we must offset our matrix so that the pivot is manually centered at gizmo origin.
						object_matrix.translate( -t )
						
						# in gizmo space, now we can rotate our object's matrix by the arcball rotation offset amounts.
						object_matrix.rotate( arcball_rotation_values )

						# now we must un-offset our matrix.
						object_matrix.translate( t )
					
					# we can just set the transform with this function.
					obj.setTransform(object_matrix)

		# logic branch for handling scaling.
		elif self.gizmo_active_action == 'scale':
			
			if self.initial_gizmo_element != 7:
				# we get the absolute point returned by our ray cast function.
				closest_3d_point = self.scale_uv_2_xyz( u , v )

				# we subtract it from what was stored in the Begin() function. this gives us relative drag, and not absolute.
				displaced_vector = closest_3d_point - self.initial_gizmo_drag

				# we want to know the distance we've dragged, but distance formula rightly produces
				# only positive values as distance is never negative. we want to know how far with sign
				# we've moved the cursor, in some dominant direction.
				# so we take an average of the offset of each axis, but before we do that, we prune out
				# any offset values near zero, so they don't weight the offset in a biased way.
				offsets = [ displaced_vector.x , displaced_vector.y , displaced_vector.z ]
				offsets = [ x for x in offsets if abs(x) > 0.001 ]
				displaced_distance = sum(offsets) / len(offsets)
			
			else:
				closest_magnitude = self.scale_uv_2_magnitude( u , v )
				displaced_distance = closest_magnitude - self.initial_gizmo_drag

			if self.gizmo_orientation_mode == 'world':

				# for each of the selected objects.
				for obj in self.affected_objects:

					# get the initial object dictionary for this object.
					objInitialParams = self.initial_object_params[obj.id]

					# get the inverse matrix of the objects parent, if it has no parent, this will be an identity matrix.
					parent_matrix = objInitialParams['parent_matrix'].copy()
					parent_matrix_inverse = objInitialParams['parent_matrix_inverse'].copy()
					object_matrix = objInitialParams['object_matrix'].copy()

					# get the initial scale values on the object.
					sx,sy,sz = objInitialParams['sx'],objInitialParams['sy'],objInitialParams['sz']

					# get the initial axis vectors of the object as they are in world space.
					vx,vy,vz = objInitialParams['vx'],objInitialParams['vy'],objInitialParams['vz']
					vx.normalize(); vy.normalize(); vz.normalize() # normalize just to be safe.
					
					# AXIS DRAG
					if self.gizmo_inverse_axis == False:
						
						gizmo_active_axis_copy = self.gizmo_active_axis.copy()
						gizmo_active_axis_copy.normalize()

						# these ratios tell us how aligned the gizmo's axis vector is with the object's vector.
						# when this is == 1, it means they are perfectly aligned, -1 would mean aligned but facing
						# in perfectly opposite directions, but that is not helpful for scale, so we take abs() 
						# in a sense this makes the vectors "directionless", 1 can mean we are aligned in either direction with abs().
						sx_ratio = abs( gizmo_active_axis_copy.dot(vx) )
						sy_ratio = abs( gizmo_active_axis_copy.dot(vy) )
						sz_ratio = abs( gizmo_active_axis_copy.dot(vz) )

						# simpler case for axis drag, we just use these ratios as is.
						x_ratio_final = sx_ratio
						y_ratio_final = sy_ratio
						z_ratio_final = sz_ratio
					
					# PLANE DRAG
					elif self.gizmo_inverse_axis == True:
						
						# make a copy of the vector, and normalize it.
						gizmo_active_axis_copy = 1 - self.gizmo_active_axis.copy()
						gizmo_active_axis_copy.normalize()

						# these ratios tell us how aligned the gizmo's axis vector is with the object's vector.
						# when this is == 1, it means they are perfectly aligned, -1 would mean aligned but facing
						# in perfectly opposite directions, but that is not helpful for scale, so we take abs() 
						# in a sense this makes the vectors "directionless", 1 can mean we are aligned in either direction with abs().
						sx_ratio = abs( gizmo_active_axis_copy.dot(vx) )
						sy_ratio = abs( gizmo_active_axis_copy.dot(vy) )
						sz_ratio = abs( gizmo_active_axis_copy.dot(vz) )

						# we're trying to figure out if any of the 3 object axes are a match to the gizmo.
						# here we check all 3, and take the maximum, this will be a value between 0-1 and 
						# provide us with a value to use to mix two other values.
						closest_axis_match = max( [
							abs( self.gizmo_active_axis.dot(vx) ),
							abs( self.gizmo_active_axis.dot(vy) ),
							abs( self.gizmo_active_axis.dot(vz) ),
						] )

						# on one end of our mix range, we want to scale the axes uniformly, but not scale the one that represents the plane normal.
						largest_ratio = max([sx_ratio,sy_ratio,sz_ratio])
						sx_ratio_unified = largest_ratio * (1-self.gizmo_active_axis.x)
						sy_ratio_unified = largest_ratio * (1-self.gizmo_active_axis.y)
						sz_ratio_unified = largest_ratio * (1-self.gizmo_active_axis.z)

						# blend between the two styles of scaling based on the closest axis match variable.
						x_ratio_final = (sx_ratio_unified*closest_axis_match) + (sx_ratio*(1-closest_axis_match))
						y_ratio_final = (sy_ratio_unified*closest_axis_match) + (sy_ratio*(1-closest_axis_match))
						z_ratio_final = (sz_ratio_unified*closest_axis_match) + (sz_ratio*(1-closest_axis_match))

					# first we must take any objects parented to other objects, and transform their matrix into world space.
					# the object_matrix will represent that object in the same transformation as before, just without a parent.
					object_matrix = parent_matrix * object_matrix
					object_world_pos = tdu.Position(object_matrix.decompose()[2])
					object_world_pos -= tdu.Vector(self.initial_gizmo_location)
					object_world_pos.x *= displaced_distance * x_ratio_final
					object_world_pos.y *= displaced_distance * y_ratio_final
					object_world_pos.z *= displaced_distance * z_ratio_final

					object_matrix.translate(object_world_pos)

					# now we've done our translation, we need to move our world space matrix back to the space it was in before.
					object_matrix = parent_matrix_inverse * object_matrix

					# we can just set the transform with this function.
					obj.setTransform(object_matrix)

					final_scale_x = sx + displaced_distance * x_ratio_final
					final_scale_y = sy + displaced_distance * y_ratio_final
					final_scale_z = sz + displaced_distance * z_ratio_final

					# if gizmo element clicked is the outer scale ring, means we need to apply uniform scale.
					if self.initial_gizmo_element == 7:
						obj.par.sx = sx + displaced_distance
						obj.par.sy = sy + displaced_distance
						obj.par.sz = sz + displaced_distance
					else:
						obj.par.sx = final_scale_x
						obj.par.sy = final_scale_y
						obj.par.sz = final_scale_z
			
			elif self.gizmo_orientation_mode == 'local':

				# for each of the selected objects.
				for obj in self.affected_objects:

					# get the initial object dictionary for this object.
					objInitialParams = self.initial_object_params[obj.id]

					# get the initial scale values on the object.
					sx,sy,sz = objInitialParams['sx'],objInitialParams['sy'],objInitialParams['sz']

					# get the gizmo's transform matrix then invert it.
					gizmo_matrix = self.ownerComp.worldTransform; gizmo_matrix.invert()

					# next take our world sapce gizmo active axis, and transform it to local space.
					# this gives us the axis x, y, or z that the user is trying to rotate around.
					gizmo_active_axis_gizmospace = gizmo_matrix * self.gizmo_active_axis
					gizmo_active_axis_gizmospace.normalize()

					# remove really small fluctuations of rotation on other axes, and set them to 0.
					gizmo_active_axis_gizmospace.x = 0 if gizmo_active_axis_gizmospace.x <= 0.001 else gizmo_active_axis_gizmospace.x
					gizmo_active_axis_gizmospace.y = 0 if gizmo_active_axis_gizmospace.y <= 0.001 else gizmo_active_axis_gizmospace.y
					gizmo_active_axis_gizmospace.z = 0 if gizmo_active_axis_gizmospace.z <= 0.001 else gizmo_active_axis_gizmospace.z

					# if user drags a plane instead of axis, inverse the gizmo masks. ie. drag plane of X means scaling YZ
					if self.gizmo_inverse_axis == True:
						gizmo_active_axis_gizmospace.x = 1 - gizmo_active_axis_gizmospace.x
						gizmo_active_axis_gizmospace.y = 1 - gizmo_active_axis_gizmospace.y
						gizmo_active_axis_gizmospace.z = 1 - gizmo_active_axis_gizmospace.z

					# if user dragged the outer ring, forget everything we did before, this means we uniformly scale all axes.
					if self.initial_gizmo_element == 7:
						gizmo_active_axis_gizmospace.x = 1
						gizmo_active_axis_gizmospace.y = 1
						gizmo_active_axis_gizmospace.z = 1

					# calculate the final scale offsets.
					final_scale_x = sx + displaced_distance * gizmo_active_axis_gizmospace.x
					final_scale_y = sy + displaced_distance * gizmo_active_axis_gizmospace.y
					final_scale_z = sz + displaced_distance * gizmo_active_axis_gizmospace.z

					# apply the final scale offsets.
					obj.par.sx = final_scale_x
					obj.par.sy = final_scale_y
					obj.par.sz = final_scale_z

				pass
		return

	def End(self):
		'''
		called when the user succesfully ends a drag function, from a renderpick callback, this is executed on event.selectEnd.
		'''
		self.Init()

		# store the orientation mode every time we initialize the gizmo.
		# self.gizmo_orientation_mode = self.ownerComp.par.Transformationorientation.eval()

		# reset to defaults.
		# self.ownerComp.par.Visible = self.ownerComp.par.Visible.default # true
		# self.ownerComp.par.Canceled = self.ownerComp.par.Canceled.default # false

		# self.update_gizmo_position()

		return

	def Cancel(self):
		'''
		If the user decides their transformation is a mistake, they can cancel the operation by calling this function. It is meant to be called by the right click
		button while inside the operational panel comp. This is designed to be called from the Gizmo comp, not the renderpick callback.
		'''

		self.ownerComp.par.Canceled = True
		self.ownerComp.Update_Picking_Status(False)

		if self.gizmo_active_action in ['translate','rotate','scale']:

			for obj in self.affected_objects:
				objInitialParams = self.initial_object_params.get( obj.id , None )
				if objInitialParams != None:
					object_matrix = objInitialParams['object_matrix'].copy()
					obj.setTransform(object_matrix)

		return