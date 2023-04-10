"""
TDFIL description
"""

import sys
sys.path.append(project.folder + 'tdfil/thirdparty/natsort')


import td
import inspect

class TDFIL:
	"""
	TDFIL description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.template_BASE = self.ownerComp.op('Templates')
		self.TypeGroup_DAT = self.ownerComp.op('TypeGroups/groups')
		self.ShaderVariant_DAT = self.ownerComp.op('shaderVariants/null_shader_variants')
		self.Td2filament_DAT = self.ownerComp.op('shaderVariants/td_2_filament')
		self.PrimitiveList_DAT = self.ownerComp.op('PrimitiveTemplates/null_prim_list')
		self.ThumbnailManager_COMP = self.ownerComp.op('ThumbnailManager')
		# self.ShaderInputs_COMP = self.ownerComp.op('ShaderInputs')
		self.SceneExt_DAT = self.ownerComp.op('sceneext')

		self.style = {
			'bgcolorr':0.05,
			'bgcolorg':0.05,
			'bgcolorb':0.05,
		}

		self.style = {
			'bgcolorr':[0.1,0.2,0.4],
			'bgcolorg':[0.1,0.2,0.4],
			'bgcolorb':[0.1,0.2,0.4],
			'bgalpha':[1.0]*3,

			'currentcolorr':[0.1,0.3,0.4],
			'currentcolorg':[0.1,0.15,0.2],
			'currentcolorb':[0.1,0.1,0.1],
			'currentalpha':[1.0]*3,

			'platecolorr':[0.05]*3,
			'platecolorg':[0.05]*3,
			'platecolorb':[0.05]*3,
			'platealpha':[1.0]*3,

			'fontcolorr':[.9]*3,
			'fontcolorg':[.9]*3,
			'fontcolorb':[.9]*3,
			'fontalpha':[1.0]*3,

			'fontminor':['Tahoma'],

			'fontpointsize':[9],
			'textpadding':[4],
			'indentsize':[20],
			'rowheight':[25],
			'rowspacing':[1],
		}

	def Style(self, par, styleState):
		'''
		par is a ref to the par. can grab this easily within a parameter expression by using me.curPar
		styleState is an integer defining what state the interaction is in. 0 is base, 1 is hover, 2 is click, etc.
		'''
		
		if isinstance(par, td.Par): # if par is a td.Par, get it's name and use that.
			par_style = self.style.get(par.name,None)
		elif isinstance(par, str): # if par is a string, we can just use it directly as is.
			par_style = self.style.get(par,None)
		else:
			debug(f'{type(par)} type not recognized... skipping.')

		if par_style == None:
			debug(f'par:{par.name} does not exist in self.style dict, must add - returning zero.')
			return 0

		styleState = int(styleState)
		try:
			return par_style[styleState]
		except:
			debug(f'style_state <{styleState}> not present in self.style dictionary, returning zero.')
			return par_style[0]

		return 
	
	@property
	def Template_ROOT(self):
		return self.template_BASE

	#### Asset Templates ####
	@property
	def Template_MaterialAsset(self):
		return self.template_BASE.op('material_asset_v3')
	
	@property
	def Template_MeshAsset(self):
		return self.template_BASE.op('mesh_asset')
	
	@property
	def Template_NullAsset(self):
		return self.template_BASE.op('null_asset')
	
	@property
	def Template_CameraAsset(self):
		return self.template_BASE.op('fila_editor_camera_asset')
	
	@property
	def Template_EnvlightAsset(self):
		return self.template_BASE.op('fila_envLight_asset')
	
	#### UI Templates ####
	@property
	def Template_UiHeader(self):
		return self.template_BASE.op('Ui_Header')
	
	@property
	def Template_UiButton(self):
		return self.template_BASE.op('Ui_Button')



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

	def TraceFunctionCall(self):
		'''
		This function will print the entire function call's 
		stack trace showing what called what where.
		Obviously if you use scriptDat.run() 
		it will break the function call chain.
		'''
		inspectedStack = inspect.stack()
		print('qwe')
		source = []
		line = []
		function = []
		for f in inspectedStack[::-1]:
			frameInfo = inspect.getframeinfo(f[0])
			source += [frameInfo[0]]
			line += [str(frameInfo[1])]
			function += [frameInfo[2]]

		source_max = max( [len(x) for x in source] )
		line_max = max( [len(x) for x in line] )
		function_max = max( [len(x) for x in function] )

		print('-'*(source_max+line_max+function_max+11))

		for i,f in enumerate(zip(source,line,function)):
			f = list(f)
			source_ = f[0].ljust(source_max)
			line_ = f[1].ljust(line_max)
			function_ = (f[2]+'()').ljust(function_max)

			print('%i) %s : %s : %s'%(i,source_,line_,function_))

		print('-'*(source_max+line_max+function_max+11))

		return
	


	def Natsort_os_sorted(self):
		pass
		return