
#if defined(MATERIAL_HAS_NORMAL)
	material.normal = materialParams_normal();
#endif

#if defined(MATERIAL_HAS_BENT_NORMAL)
	material.bentNormal = materialParams_bentNormal();
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
	material.clearCoatNormal = materialParams_clearCoatNormal();
#endif

prepareMaterial(material);

// always on.
#if defined(MATERIAL_HAS_BASE_COLOR)
	material.baseColor.rgb = materialParams_baseColor();
#endif

#if defined(BLEND_MODE_TRANSPARENT) || defined(BLEND_MODE_MASKED)
	material.baseColor.a = materialParams_alpha();
	material.baseColor.rgb *= materialParams_alpha();
#endif

#if defined(MATERIAL_HAS_METALLIC)
	material.metallic = materialParams_metallic();
#endif

#if defined(MATERIAL_HAS_ROUGHNESS)
	material.roughness = materialParams_roughness();
#endif

#if defined(MATERIAL_HAS_REFLECTANCE)
	material.reflectance = materialParams_reflectance();
#endif

#if defined(MATERIAL_HAS_SHEEN_COLOR)
	material.sheenColor = materialParams_sheenColor();
	
#endif

#if defined(MATERIAL_HAS_SHEEN_ROUGHNESS)
	material.sheenRoughness = materialParams_sheenRoughness();
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
	material.clearCoat = materialParams_clearCoat();
	material.clearCoatRoughness = materialParams_clearCoatRoughness();
	#if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
		material.clearCoatNormal = materialParams_clearCoatNormal();
	#endif
#endif

#if defined(MATERIAL_HAS_ANISOTROPY)
	material.anisotropy = materialParams_anisotropy();
	material.anisotropyDirection = materialParams_anisotropyDirection();
#endif

#if defined(MATERIAL_HAS_AMBIENT_OCCLUSION)
	material.ambientOcclusion = materialParams_ambientOcclusion();
#endif

#if defined(MATERIAL_HAS_IOR)
	material.ior = materialParams_ior();
#endif

#if defined(MATERIAL_HAS_EMISSIVE)
	material.emissive = materialParams_emissive();
#endif

#if defined(MATERIAL_HAS_POST_LIGHTING_COLOR)
	material.postLightingColor = materialParams_postLightingColor();
#endif

// material.alpha = materialParams_alpha();

#if defined(MATERIAL_HAS_TRANSMISSION) 
	material.transmission = materialParams_transmission();
#endif

#if defined(MATERIAL_HAS_ABSORPTION)
	material.absorption = materialParams_absorption();
#endif

#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
	material.microThickness = materialParams_microThickness();
#endif

#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
	material.thickness = materialParams_thickness();
#endif

#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
	material.subsurfaceColor = materialParams_subsurfaceColor();
#endif

#if defined(MATERIAL_HAS_SUBSURFACE_POWER)
	material.subsurfacePower = materialParams_subsurfacePower();
#endif