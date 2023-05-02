def define_FILAMENT_QUALITY(scriptOp):
	scriptOp.text += '''
// filament quality, for desktop should likely always be high.
#define FILAMENT_QUALITY_LOW    0
#define FILAMENT_QUALITY_NORMAL 1
#define FILAMENT_QUALITY_HIGH   2
#define FILAMENT_QUALITY FILAMENT_QUALITY_HIGH
'''
	return

def define_SPECULAR_AMBIENT_OCCLUSION(scriptOp):
	# if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
	scriptOp.text += '''
// filament specular AO quality, 2 would technically be better, but it's bugged or my implementation is wrong, so we use quality lvl 1.
#define SPECULAR_AMBIENT_OCCLUSION 1
'''
	return

def define_MULTI_BOUNCE_AMBIENT_OCCLUSION(scriptOp):
	# if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
	scriptOp.text += '''
// for desktop 1 is ideal.
#define MULTI_BOUNCE_AMBIENT_OCCLUSION 1
'''
	return

def define_GEOMETRIC_SPECULAR_AA(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
		if parent.obj.par.Enablespecularantialiasing.eval() == True:
			scriptOp.text += f'\n//on or off, for desktop, on is ideal.\n'
			scriptOp.text += f'#define GEOMETRIC_SPECULAR_AA\n'
	return

def define_SHADING_MODEL(scriptOp):
	scriptOp.text += f'\n//blend mode: 0=unlit, 1=lit, 2=subsurface, 3=cloth \n'
	if parent.obj.par.Shadingmodel.eval() == 'unlit':
		scriptOp.text += f'#define SHADING_MODEL_UNLIT \n'
	if parent.obj.par.Shadingmodel.eval() == 'lit':
		scriptOp.text += f'#define SHADING_MODEL_LIT \n'
	if parent.obj.par.Shadingmodel.eval() == 'subsurface':
		scriptOp.text += f'#define SHADING_MODEL_SUBSURFACE \n'
	if parent.obj.par.Shadingmodel.eval() == 'cloth':
		scriptOp.text += f'#define SHADING_MODEL_CLOTH \n'
	return

def define_BLENDING_MODE(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['unlit', 'subsurface', 'cloth']:
		scriptOp.text += f'#define BLEND_MODE_OPAQUE \n'
	elif parent.obj.par.Shadingmodel.eval() == 'lit':
		if parent.obj.par.Blendingmodel.eval() == 'opaque':
			scriptOp.text += f'#define BLEND_MODE_OPAQUE \n'
		elif parent.obj.par.Blendingmodel.eval() == 'transparent':
			scriptOp.text += f'#define BLEND_MODE_TRANSPARENT \n'
		elif parent.obj.par.Blendingmodel.eval() == 'fade':
			scriptOp.text += f'#define BLEND_MODE_TRANSPARENT \n'
			scriptOp.text += f'#define BLEND_MODE_FADE \n'
		elif parent.obj.par.Blendingmodel.eval() in ['thinrefraction', 'solidrefraction']:
			scriptOp.text += f'#define MATERIAL_HAS_REFRACTION \n'
			scriptOp.text += f'#define REFRACTION_MODE_CUBEMAP 1 \n'
			scriptOp.text += f'#define REFRACTION_MODE_SCREEN_SPACE 2 \n'
			if parent.obj.par.Screenspacerefraction == True:
				scriptOp.text += f'#define REFRACTION_MODE REFRACTION_MODE_SCREEN_SPACE \n'
			else:
				scriptOp.text += f'#define REFRACTION_MODE REFRACTION_MODE_CUBEMAP \n'
			scriptOp.text += f'#define REFRACTION_TYPE_SOLID 0 \n'
			scriptOp.text += f'#define REFRACTION_TYPE_THIN 1 \n'
			if parent.obj.par.Blendingmodel.eval() == 'solidrefraction':
				scriptOp.text += f'#define REFRACTION_TYPE REFRACTION_TYPE_SOLID \n'
			elif parent.obj.par.Blendingmodel.eval() == 'thinrefraction':
				scriptOp.text += f'#define REFRACTION_TYPE REFRACTION_TYPE_THIN \n'
			scriptOp.text += f'#define MATERIAL_HAS_ABSORPTION \n'
			scriptOp.text += f'#define MATERIAL_HAS_TRANSMISSION \n'
			scriptOp.text += f'#define MATERIAL_HAS_IOR \n'
			if parent.obj.par.Blendingmodel.eval() == 'thinrefraction':
				scriptOp.text += f'#define MATERIAL_HAS_MICRO_THICKNESS \n'
			elif parent.obj.par.Blendingmodel.eval() == 'solidrefraction':
				scriptOp.text += f'#define MATERIAL_HAS_THICKNESS \n'
			scriptOp.text += f'#define THICKNESS_SOURCE_{parent.obj.par.Thicknesssource.menuIndex} \n'
			if parent.obj.par.Thicknesssource.menuIndex == 1:
				scriptOp.text += f'uniform sampler2D mat_thickness; \n'
		else:
			debug(f'unknown blend mode: {parent.obj.par.Blendingmodel.eval()}')
	return




def define_BASE_COLOR(scriptOp):
	scriptOp.text += f'\n//baseColor: 0=uniform, 1=top, 2=uniform+top, 3=uniform*top \n'
	scriptOp.text += f'#define BASE_COLOR_METHOD_{parent.obj.par.Basecolormethod.menuIndex} \n'
	if parent.obj.par.Basecolormethod.menuIndex in [1, 2, 3]:
		scriptOp.text += f'uniform sampler2D mat_baseColor; \n'
	scriptOp.text += '#define MATERIAL_HAS_BASE_COLOR\n'
	scriptOp.text += '\n'
	return

def define_NORMAL(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
		scriptOp.text += f'//normal: 0=none, 1=top \n'
		scriptOp.text += f'#define NORMAL_METHOD_{parent.obj.par.Normalmethod.menuIndex} \n'
		if parent.obj.par.Normalmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_normal; \n'
			scriptOp.text += f'#define NORMAL_STYLE_{parent.obj.par.Normalstyle.menuIndex} // 0=opengl, 1=directx \n'
			scriptOp.text += f'#define MATERIAL_HAS_NORMAL\n'
		scriptOp.text += '\n'
	return

def define_BENT_NORMAL(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
		scriptOp.text += f'//bentNormal: 0=none, 1=top \n'
		scriptOp.text += f'#define BENT_NORMAL_METHOD_{parent.obj.par.Bentnormalmethod.menuIndex} \n'
		if parent.obj.par.Bentnormalmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_bentNormal; \n'
			scriptOp.text += f'#define BENT_NORMAL_STYLE_{parent.obj.par.Bentnormalstyle.menuIndex} // 0=opengl, 1=directx\n'
			scriptOp.text += f'#define MATERIAL_HAS_BENT_NORMAL\n'
		scriptOp.text += '\n'
	return

def define_AMBIENT_OCCULSION(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
		scriptOp.text += f'//ambientOcclusion: 0=none, 1=top \n'
		scriptOp.text += f'#define AMBIENT_OCCLUSION_METHOD_{parent.obj.par.Ambientocclusionmethod.menuIndex} \n'
		if parent.obj.par.Ambientocclusionmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_ambientOcclusion; \n'
			scriptOp.text += f'#define MATERIAL_HAS_AMBIENT_OCCLUSION\n'
		if parent.obj.par.Globalssao.menuIndex in [1]:
			scriptOp.text += f'#define MATERIAL_HAS_SSAO\n'
		scriptOp.text += '\n'
	
	# if parent.obj.par.Shadingmodel.eval() in ['unlit']:
	# 	scriptOp.text += f'#define MATERIAL_HAS_AMBIENT_OCCLUSION\n'
	# 	scriptOp.text += '\n'
	return

def define_METALLIC(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface']:
		scriptOp.text += f'//metalness: 0=uniform, 1=top \n'
		scriptOp.text += f'#define METALNESS_METHOD_{parent.obj.par.Metalnessmethod.menuIndex} \n'
		if parent.obj.par.Metalnessmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_metalness; \n'
		scriptOp.text += '#define MATERIAL_HAS_METALLIC\n'
		scriptOp.text += '\n'
	return

def define_ROUGHNESS(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
		scriptOp.text += f'//roughness: 0=uniform, 1=top \n'
		scriptOp.text += f'#define ROUGHNESS_METHOD_{parent.obj.par.Roughnessmethod.menuIndex} \n'
		if parent.obj.par.Roughnessmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_roughness; \n'
		scriptOp.text += '#define MATERIAL_HAS_ROUGHNESS\n'
		scriptOp.text += '\n'
	return

def define_REFLECTANCE(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface']:
		if parent.obj.par.Blendingmodel.eval() in ['opaque','transparent','fade']:
			scriptOp.text += f'//reflectance: 0=uniform, 1=top \n'
			scriptOp.text += f'#define REFLECTANCE_METHOD_{parent.obj.par.Reflectancemethod.menuIndex} \n'
			if parent.obj.par.Reflectancemethod.menuIndex in [1]:
				scriptOp.text += f'uniform sampler2D mat_reflectance; \n'
			scriptOp.text += '#define MATERIAL_HAS_REFLECTANCE\n'
			scriptOp.text += '\n'
	return

def define_CLEARCOAT(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit']:
		scriptOp.text += f'//clearcoat: 0=none, 1=uniform, 2=top \n'
		scriptOp.text += f'#define CLEAR_COAT_METHOD_{parent.obj.par.Clearcoatmethod.menuIndex} \n'
		if parent.obj.par.Clearcoatmethod.menuIndex > 0:
			scriptOp.text += f'#define MATERIAL_HAS_CLEAR_COAT\n'
			scriptOp.text += f'#define MATERIAL_HAS_CLEAR_COAT_ROUGHNESS\n'
			if parent.obj.par.Clearcoatmethod.menuIndex in [2]:
				scriptOp.text += f'uniform sampler2D mat_clearCoat; \n'
			scriptOp.text += f'#define CLEAR_COAT_NORMAL_METHOD_{parent.obj.par.Clearcoatnormalmethod.menuIndex} \n'
			if parent.obj.par.Clearcoatnormalmethod.menuIndex in [1]:
				scriptOp.text += f'#define MATERIAL_HAS_CLEAR_COAT_NORMAL\n'
				scriptOp.text += f'uniform sampler2D mat_clearCoatNormal; \n'
			scriptOp.text += f'#define CLEAR_COAT_NORMAL_STYLE_{parent.obj.par.Clearcoatnormalstyle.menuIndex} // 0=opengl, 1=directx\n'
			if parent.obj.par.Clearcoatchangesior == True:
				scriptOp.text += f'#define CLEAR_COAT_IOR_CHANGE\n'
			scriptOp.text += f'#define CLEAR_COAT_ROUGHNESS_METHOD_{parent.obj.par.Clearcoatroughnessmethod.menuIndex} \n'
			if parent.obj.par.Clearcoatroughnessmethod.menuIndex in [1]:
				scriptOp.text += f'uniform sampler2D mat_clearCoatRoughness; \n'
		scriptOp.text += '\n'
	return

def define_SHEEN(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'cloth']:
		scriptOp.text += f'//sheen: 0=none, 1=uniform, 2=top \n'
		scriptOp.text += f'#define SHEEN_METHOD_{parent.obj.par.Sheenmethod.menuIndex} \n'
		if parent.obj.par.Sheenmethod.menuIndex > 0 or parent.obj.par.Shadingmodel.eval() == 'cloth':
			if parent.obj.par.Sheenmethod.menuIndex in [2]:
				scriptOp.text += f'uniform sampler2D mat_sheenColor; \n'
			if parent.obj.par.Shadingmodel.eval() != 'cloth':
				scriptOp.text += f'#define MATERIAL_HAS_SHEEN_COLOR\n'
			if parent.obj.par.Shadingmodel.eval() in ['lit']:
				scriptOp.text += f'#define MATERIAL_HAS_SHEEN_ROUGHNESS\n'
			scriptOp.text += f'#define SHEEN_ROUGHNESS_METHOD_{parent.obj.par.Sheenroughnessmethod.menuIndex} \n'
		if parent.obj.par.Sheenroughnessmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_sheenRoughness; \n'
		scriptOp.text += '\n'
	return

def define_ANISOTROPY(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit']:
		scriptOp.text += f'//anisotropy: 0=none, 1=uniform, 2=top \n'
		scriptOp.text += f'#define ANISOTROPY_METHOD_{parent.obj.par.Anisotropymethod.menuIndex} \n'
		if parent.obj.par.Anisotropymethod.menuIndex > 0:
			if parent.obj.par.Anisotropymethod.menuIndex in [2]:
				scriptOp.text += f'uniform sampler2D mat_anisotropy; \n'
			scriptOp.text += f'#define MATERIAL_HAS_ANISOTROPY\n'
		scriptOp.text += f'#define MATERIAL_HAS_ANISOTROPY_DIRECTION\n'
		scriptOp.text += f'#define ANISOTROPY_DIRECTION_METHOD_{parent.obj.par.Anisotropydirectionmethod.menuIndex} \n'
		if parent.obj.par.Anisotropydirectionmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_anisotropyDirection; \n' 
		scriptOp.text += '\n'
	return

def define_SUBSURFACE(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['subsurface', 'cloth']:
		if parent.obj.par.Shadingmodel.eval() in ['subsurface','cloth']:
			scriptOp.text += f'//subsurface color: 0=uniform, 1=top \n'
			scriptOp.text += f'#define SUBSURFACE_COLOR_METHOD_{parent.obj.par.Subsurfacecolormethod.menuIndex} \n'
			scriptOp.text += f'#define MATERIAL_HAS_SUBSURFACE_COLOR\n'
			if parent.obj.par.Subsurfacecolormethod.menuIndex in [1]:
				scriptOp.text += f'uniform sampler2D mat_subsurfaceColor; \n'
		if parent.obj.par.Shadingmodel.eval() in ['subsurface']:
			scriptOp.text += f'#define MATERIAL_HAS_SUBSURFACE_POWER\n'
		scriptOp.text += f'//subsurface thickness: 0=uniform, 1=top*uniform \n'
		scriptOp.text += f'#define SUBSURFACE_THICKNESS_SOURCE_{parent.obj.par.Subsurfacethicknessmethod.menuIndex} \n'
		scriptOp.text += f'#define MATERIAL_HAS_THICKNESS\n'
		if parent.obj.par.Subsurfacethicknessmethod.menuIndex in [1]:
			scriptOp.text += f'uniform sampler2D mat_subsurfaceThickness; \n'

		
		scriptOp.text += '\n'
	return

def define_MATERIAL_NEEDS_TBN(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
	# 	has_normals = scriptOp.text.find('#define MATERIAL_HAS_NORMAL') > -1
	# 	has_bent_normals = scriptOp.text.find('#define MATERIAL_HAS_BENT_NORMAL') > -1
	# 	has_clearcoat_normals = scriptOp.text.find('#define MATERIAL_HAS_CLEAR_COAT_NORMAL') > -1
	# 	mat_needs_tbn = has_normals or has_bent_normals or has_clearcoat_normals
	# 	if mat_needs_tbn == True:
		scriptOp.text += f'#define MATERIAL_NEEDS_TBN \n'
	return

def define_MATERIAL_HAS_EMISSIVE(scriptOp):
	if parent.obj.par.Emissivemethod.eval() in ['uniform','top']:
		scriptOp.text += f'//subsurface color: 0=none, 1=uniform, 2=top \n'
		scriptOp.text += f'#define EMISSIVE_METHOD_{parent.obj.par.Emissivemethod.menuIndex} \n'
		scriptOp.text += f'#define MATERIAL_HAS_EMISSIVE\n'
		if parent.obj.par.Emissivemethod.eval() in ['top']:
			scriptOp.text += f'uniform sampler2D mat_emissive; \n'
		scriptOp.text += '\n'
	return

def define_MATERIAL_HAS_POST_LIGHTING(scriptOp):
	if parent.obj.par.Postlightingmethod.eval() in ['uniform', 'top']:
		blend_options = \
		['POST_LIGHTING_BLEND_MODE_OPAQUE',
		'POST_LIGHTING_BLEND_MODE_TRANSPARENT',
		'POST_LIGHTING_BLEND_MODE_ADD',
		'POST_LIGHTING_BLEND_MODE_MULTIPLY',
		'POST_LIGHTING_BLEND_MODE_SCREEN']
		scriptOp.text += f'//subsurface color: 0=none, 1=uniform, 2=top \n'
		scriptOp.text += f'#define POST_LIGHTING_METHOD_{parent.obj.par.Postlightingmethod.menuIndex} \n'
		scriptOp.text += f'#define MATERIAL_HAS_POST_LIGHTING_COLOR\n'
		if parent.obj.par.Postlightingmethod.eval() in ['top']:
			scriptOp.text += f'uniform sampler2D mat_postLightingColor; \n'
		scriptOp.text += f'#define {blend_options[parent.obj.par.Postlightingblendmode.menuIndex]} \n'
		scriptOp.text += '\n'
	return

def define_SHADING_INTERPOLATION(scriptOp):
	scriptOp.text += f'#define SHADING_INTERPOLATION\n'
	scriptOp.text += '\n'
	return

def define_HAS_ATTRIBUTE_TANGENTS(scriptOp):
	if parent.obj.par.Shadingmodel.eval() in ['lit', 'subsurface', 'cloth']:
		scriptOp.text += f'#define HAS_ATTRIBUTE_TANGENTS\n'
		scriptOp.text += '\n'
	return

def define_MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY(scriptOp):
	if parent.obj.par.Doublesided.eval() in ['yes']:
		scriptOp.text += f'#define MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY\n'
		scriptOp.text += '\n'
	return

def define_POST_LIGHTING_BLEND_MODE_TRANSPARENT(scriptOp):
	scriptOp.text += f'#define POST_LIGHTING_BLEND_MODE_TRANSPARENT\n'
	scriptOp.text += '\n'
	return

def define_HAS_ATTRIBUTE_UV0(scriptOp):
	scriptOp.text += f'#define HAS_ATTRIBUTE_UV0 \n'
	scriptOp.text += '\n'	

def define_HAS_ATTRIBUTE_COLOR(scriptOp):
	# if we need to, we can bring back explicit control of vertex color
	# if parent.obj.par.Vertexcolormethod.eval() in ['enabled']:
	scriptOp.text += f'#define HAS_ATTRIBUTE_COLOR \n'
	scriptOp.text += '\n'	

def define_VARIANT_LIGHTING(scriptOp):
	scriptOp.text += f'#define VARIANT_HAS_DIRECTIONAL_LIGHTING\n'
	scriptOp.text += f'#define VARIANT_HAS_SHADOWING\n'
	scriptOp.text += f'#define VARIANT_HAS_DYNAMIC_LIGHTING\n'
	scriptOp.text += '\n'

	

def onCook(scriptOp):

	# get input text dat which also conveninently sets this script DAT to text style.
	scriptOp.copy(scriptOp.inputs[0])
	
	define_FILAMENT_QUALITY(scriptOp)

	define_SPECULAR_AMBIENT_OCCLUSION(scriptOp)
	
	define_MULTI_BOUNCE_AMBIENT_OCCLUSION(scriptOp)

	define_GEOMETRIC_SPECULAR_AA(scriptOp)

	define_SHADING_MODEL(scriptOp)

	define_BLENDING_MODE(scriptOp)

	define_BASE_COLOR(scriptOp)

	define_NORMAL(scriptOp)

	define_BENT_NORMAL(scriptOp)

	define_AMBIENT_OCCULSION(scriptOp)

	define_METALLIC(scriptOp)

	define_ROUGHNESS(scriptOp)

	define_REFLECTANCE(scriptOp)

	define_CLEARCOAT(scriptOp)

	define_SHEEN(scriptOp)

	define_ANISOTROPY(scriptOp)

	define_SUBSURFACE(scriptOp)

	define_MATERIAL_HAS_EMISSIVE(scriptOp)

	define_MATERIAL_HAS_POST_LIGHTING(scriptOp)

	define_MATERIAL_NEEDS_TBN(scriptOp)
	define_SHADING_INTERPOLATION(scriptOp)
	define_HAS_ATTRIBUTE_TANGENTS(scriptOp)
	define_MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY(scriptOp)
	# define_POST_LIGHTING_BLEND_MODE_TRANSPARENT(scriptOp)
	define_HAS_ATTRIBUTE_UV0(scriptOp)
	define_HAS_ATTRIBUTE_COLOR(scriptOp)

	# define_VARIANT_LIGHTING(scriptOp)


	scriptOp.text += '\n'

	



	return
