import td
import inspect # this might not be neccesary if running inside of TD.

class EditorExt:
	"""
	EditorExt description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.Viewport_COMP = self.ownerComp.op('Viewport')
		self.Renderer_COMP = self.ownerComp.op('Viewport/Renderer')

		# initialize container to sensible default size.
		self.ownerComp.par.w = 1280
		self.ownerComp.par.h = 720
		

	@property
	def OutlinerWrapper(self):
		return self.ownerComp.op('container_outliner_wrapper')

	@property
	def Outliner(self):
		return self.ownerComp.op('outliner')

	@property
	def AttributeEditor(self):
		return self.ownerComp.op('container_UG_V4')
	
	@property
	def Scene(self):
		return self.ownerComp.par.Scenecomp.eval()
	
	@property
	def Camera(self):
		return self.ownerComp.op('Viewport/Camera')
	
	@property
	def TransformGizmo(self):
		return self.ownerComp.op('Viewport/TransformGizmo')
	
	@property
	def InstancePainterGizmo(self):
		return self.ownerComp.op('Viewport/InstancePainterGizmo')
	
	@property
	def SettingsManager(self):
		return self.ownerComp.op('Viewport/Renderer/SettingsManager')
	
	@property
	def Scene(self):
		return self.ownerComp.par.Scenecomp.eval()
	
	@property
	def Template_ROOT(self):
		return op.TDFIL.Template_ROOT
	
	def Create_Object(self, Objtype):
		f = self.Template_ROOT.findChildren(parName='Objtype')
		f = [ x for x in f if x.par.Objtype.eval() == Objtype ]

		if len(f) == 0:
			print(f'No template objects had an Objtype parameter value of <{Objtype}, cannot proceed...>')
			return

		elif len(f) > 1:
			print(f'More than one template object had an Objtype parameter value of <{Objtype}, cannot proceed...>')
			return
		
		else:
			template = f[0]
			f = self.Scene.copyOPs([ template ])
			self.Scene.Select_Objects(f, clear_previous=True)
			return f
	

	def New_Scene(self, toxpath=None):
		'''
		wipes the contents of the scene to start over.
		'''
		self.Scene.Delete_All()

		# reset the camera object to default settings.
		self.Camera.par.reinitextensions.pulse()
		self.Camera.Look_At()
		for par in self.Camera.customPars:
			if par.page.name in ['GENERAL', 'CONTROL']:
				if par.readOnly == False and par.mode == ParMode.CONSTANT:
					par.val = par.default

		# create environment light.
		envlight_obj = self.Create_Object( self.Type_Group('ENVLIGHT')[0] )[0]
		envlight_obj.nodeX , envlight_obj.nodeY = 0 , 500

		# create default material.
		material_obj = self.Create_Object( self.Type_Group('MATERIAL')[0] )[0]
		material_obj.nodeX , material_obj.nodeY = 0 , 0

		# create default mesh.
		manlowpoly_obj = self.Create_Object( self.Type_Group('MESH')[0] )[0]
		manlowpoly_obj.nodeX , manlowpoly_obj.nodeY = 0 , -200

		# configure manlowpoly mesh
		manlowpoly_obj.op('reset_parameters').run()
		manlowpoly_obj.par.Materialcomp = material_obj.name
		manlowpoly_obj.par.Source = 'primitive'
		manlowpoly_obj.par.Primitivetype = 'man_lowpoly'
		manlowpoly_obj.name = 'mesh_lowpolyman'
		manlowpoly_obj.par.ty = 0
		manlowpoly_obj.par.tx = -.15

		# configure default material look
		material_obj.op('reset_parameters').run()
		material_obj.par.Basecolorr = 0.6
		material_obj.par.Basecolorg = 0.6
		material_obj.par.Basecolorb = 0.6
		material_obj.par.Metalstrength = 0
		material_obj.par.Roughnessstrength = 0.4
		
		# create floor mesh.
		floormesh_obj = self.Create_Object( self.Type_Group('MESH')[0] )[0]
		floormesh_obj.nodeX , floormesh_obj.nodeY = 200 , -200

		# configure floor mesh
		floormesh_obj.op('reset_parameters').run()
		floormesh_obj.par.Materialcomp = material_obj.name
		floormesh_obj.par.Source = 'primitive'
		floormesh_obj.par.Primitivetype = 'plane_1m'
		floormesh_obj.name = 'mesh_floor'
		floormesh_obj.par.scale = 10
		
		# create sphere mesh.
		spheremesh_obj = self.Create_Object( self.Type_Group('MESH')[0] )[0]
		spheremesh_obj.nodeX , spheremesh_obj.nodeY = -200 , -200

		# configure sphere mesh
		spheremesh_obj.op('reset_parameters').run()
		spheremesh_obj.par.Materialcomp = material_obj.name
		spheremesh_obj.par.Source = 'primitive'
		spheremesh_obj.par.Primitivetype = 'torus_1m'
		spheremesh_obj.name = 'mesh_sphere'
		spheremesh_obj.par.scale = 2
		spheremesh_obj.par.ty = 0
		

		return
	
	def Save_Scene(self, toxpath=None):
		'''
		saves out the scene as a tox.
		'''

		SceneCOMP = self.Scene
		SceneCOMP.store('Camera',self.Camera.Get_Data())
		SceneCOMP.store('Settings',self.SettingsManager.Get_Data())

		if toxpath == None:
			# no tox path provided, we launch file dialogue.
			toxpath = ui.chooseFile(load=False,fileTypes=['tox'],title='Save scene as:')
		
		if toxpath != None:
			SceneCOMP.save(toxpath)

		else:
			debug('toxpath ended up being None, skipping save procedure...')

		return



	def Load_Scene(self, toxpath=None):
		'''
		saves out the scene as a tox.
		'''

		SceneCOMP = self.Scene

		if toxpath == None:
			# no tox path provided, we launch file dialogue.
			toxpath = ui.chooseFile(load=True,fileTypes=['tox'],title='Choose a scene to load')
		
		if toxpath != None:
			SceneCOMP.par.externaltox = toxpath
			SceneCOMP.par.reinitnet.pulse()

			self.Camera.Set_Data( SceneCOMP.fetch('Camera',None) )
			self.SettingsManager.Set_Data( SceneCOMP.fetch('Settings',None) )

		else:
			debug('toxpath ended up being None, skipping load procedure...')

		return
	
	def Delete_Selected(self):
		self.Scene.Delete_Selected()

	def Duplicate_Selected(self):
		self.Scene.Duplicate_Selected()

	def Parent_Selected_To_Current(self):
		self.Scene.Parent_Selected_To_Current()

	def Unparent_Selected(self):
		self.Scene.Unparent_Selected()
	
	def Select_Immediate_Parents(self):
		self.Scene.Select_Immediate_Parents()
	
	def Select_Immediate_Children(self):
		self.Scene.Select_Immediate_Children()

	def Deselect_All_Objects(self):
		self.Scene.Deselect_All_Objects()

	def Deselect_All_Instances(self):
		self.Scene.Deselect_All_Instances()