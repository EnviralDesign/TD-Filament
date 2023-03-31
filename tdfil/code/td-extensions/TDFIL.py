"""
TDFIL description
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
		self.TypeGroup_DAT = self.template_BASE.op('type_groups/groups')
		self.ShaderVariant_DAT = self.template_BASE.op('shader_variants/null_shader_variants')
	
	@property
	def Template_ROOT(self):
		return self.template_BASE

	@property
	def Template_MaterialAsset(self):
		return self.template_BASE.op('material_asset_v3')
	
	@property
	def Template_MeshAsset(self):
		return self.template_BASE.op('mesh_asset')
	
	@property
	def Template_NullAsset(self):
		return self.template_BASE.op('null_asset')
	




	def Type_Group(self, typeGroupStr=''):
		'''
		does a lookup into the type group dat, and returns a list of matching Objtype ints that match that group.
		'''

		if typeGroupStr == '':
			return [ x.val for x in self.TypeGroup_DAT.col('_id_')[2::] if x.val not in ['']]

		typeGroupRow = self.TypeGroup_DAT.row(typeGroupStr)
		if typeGroupRow == None:
			debug(f'<{typeGroupStr}> is not a valid type group name. Please update or refer to {self.TypeGroup_DAT}')
			return
		cols = [x.col for x in typeGroupRow[1::] if x.val not in ['','0']]
		ids = [ int(each.val) for each in self.TypeGroup_DAT.row('_id_') if each.col in cols ]

		return ids
	
	def Shader_Variant(self, variant_name=''):
		'''
		returns a list of shader variants that match the given name.
		'''

		# if no variant name is given, return all shader variant names.
		if variant_name == '':
			return [ x.val for x in self.ShaderVariant_DAT.col('shader_name')[1::] if x.val not in ['']]

		# if a variant name is given, return the matching variant name.
		filtered_shader_variants = [ x.val for x in self.ShaderVariant_DAT.col('shader_name')[1::] if x.val == variant_name]
		if len(filtered_shader_variants) != 1: # if the variant name doesn't match exactly 1 variant, return None.
			debug('variant_name must match exactly 1 variant...')
			return None
		
		return filtered_shader_variants[0]