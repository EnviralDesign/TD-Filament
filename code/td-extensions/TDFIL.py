"""
Extension classes enhance TouchDesigner components with python. An
extension is accessed via ext.ExtensionClassName from any operator
within the extended component. If the extension is promoted via its
Promote Extension parameter, all its attributes with capitalized names
can be accessed externally, e.g. op('yourComp').PromotedFunction().

Help: search "Extensions" in wiki
"""

from TDStoreTools import StorageManager
import TDFunctions as TDF

class TDFIL:
	"""
	TDFIL description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.template_BASE = self.ownerComp.op('Templates')

	@property
	def Template_MaterialAsset(self):
		return self.template_BASE.op('material_asset_v3')
	
	@property
	def Template_MeshAsset(self):
		return self.template_BASE.op('mesh_asset')
	
	@property
	def Template_NullAsset(self):
		return self.template_BASE.op('null_asset')