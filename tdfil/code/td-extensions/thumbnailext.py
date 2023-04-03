import numpy

class thumbnailext:
	"""
	thumbnailext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.materialThumbCOMP = op('material_thumb')
		self.materialBgTOP = op('material_thumb/bg')

		self.meshThumbCOMP = op('mesh_thumb')
		self.meshBgTOP = op('mesh_thumb/bg')

		self.iblThumbTOP = op('ibl_thumb/bg')
		self.pointThumbTOP = op('pointlight_thumb/bg')
		self.spotThumbTOP = op('spotlight_thumb/bg')

		self.ThumbnailCOMPS = {
			0:op('mesh_thumb'), # mesh
			1:op('material_thumb'), # material
			2:op('null_thumb'), # null
			10:op('ibl_thumb'), # envlight
			11:op('pointlight_thumb'), # pointlight
			12:op('spotlight_thumb'), # spotlight
			100:op('instance_thumb'), # instance node
			202:op('camera_thumb'), # camera
			220:op('settingsmanager_thumb'), # settingsmanager node
			221:op('lightmanager_thumb'), # lightmanager node
		}

	def Fetch_Outliner_Thumb_Path(self, sourceAssetCOMP):
		Objtype = sourceAssetCOMP.par.Objtype.eval()
		Obj = self.ThumbnailCOMPS.get(Objtype,None)

		if Obj == None:
			debug(f'{sourceAssetCOMP} not found in thumbnail creation modules, skipping...')
			return None
		
		return Obj.op('outliner')


	def Create_Thumbnail(self, destinationScriptTOP, sourceAssetCOMP, resolution):
		'''
		A catch all function for returning a thumbnail of an object who's thumbnail is not dynamic.
		The Objtype parameter determins the texture that is returned.
		'''

		Objtype = sourceAssetCOMP.par.Objtype.eval()
		Obj = self.ThumbnailCOMPS.get(Objtype,None)

		if Obj == None:
			debug(f'{sourceAssetCOMP} not found in thumbnail creation modules, skipping...')
			return

		Obj.par.Size = resolution
		Obj.par.Comp = sourceAssetCOMP
		Bg = Obj.op('bg')
		Bg.cook(force=True)

		img = Bg.numpyArray(delayed=False, writable=True)

		# numpy image array comes from top as 32 bit always, covnert to 8 to save space.
		img *= 255
		img = img.astype(numpy.uint8)
		# img[0::, 0::, 3] = 255 # set alpha to 1

		# print(destinationScriptTOP)

		destinationScriptTOP.copyNumpyArray(img)
		destinationScriptTOP.cook(force=True)