


def onCook(scriptOp):
	scriptOp.clear()

	materialParams = op.TDF.MaterialParamsTemplate()

	materialParams.Set_baseColor(
		parent.obj.par.Basecoloruniformr.eval(),
		parent.obj.par.Basecoloruniformg.eval(),
		parent.obj.par.Basecoloruniformb.eval()
	)

	materialParams.Set_alpha( parent.obj.par.Blendingalpha.eval() )

	materialParams.Set_metallic( parent.obj.par.Metalnessfactor.eval() )
	materialParams.Set_roughness( parent.obj.par.Roughnessfactor.eval() )
	materialParams.Set_reflectance( parent.obj.par.Reflectancefactor.eval() )
	materialParams.Set_clearcoat( parent.obj.par.Clearcoatfactor.eval() )
	materialParams.Set_clearcoatRoughness( parent.obj.par.Clearcoatroughnessfactor.eval() )
	materialParams.Set_clearCoatNormalStrength( parent.obj.par.Clearcoatnormalstrength.eval() )
	materialParams.Set_anisotropy( parent.obj.par.Anisotropyfactor.eval() )
	materialParams.Set_anisotropyDirection(
		parent.obj.par.Anisotropydirectionx.eval(),
		parent.obj.par.Anisotropydirectiony.eval(),
		parent.obj.par.Anisotropydirectionz.eval()
	)
	materialParams.Set_ambientOcclusion( parent.obj.par.Ambientocclusionweight.eval() )
	materialParams.Set_normalStrength( parent.obj.par.Normalstrength.eval() )

	materialParams.Set_emissive(
		parent.obj.par.Emissivecoloruniformr.eval(),
		parent.obj.par.Emissivecoloruniformg.eval(),
		parent.obj.par.Emissivecoloruniformb.eval(),
		parent.obj.par.Emissiveev.eval(),
		parent.obj.par.Emissiveexposureweight.eval()
	)

	materialParams.Set_postLightingColor(
		parent.obj.par.Postlightingcoloruniformr.eval(),
		parent.obj.par.Postlightingcoloruniformg.eval(),
		parent.obj.par.Postlightingcoloruniformb.eval(),
		parent.obj.par.Postlightingcoloruniforma.eval()
	)

	materialParams.Set_sheenColor(
		parent.obj.par.Sheencolorr.eval(),
		parent.obj.par.Sheencolorg.eval(),
		parent.obj.par.Sheencolorb.eval()
	)
	materialParams.Set_sheenRoughness( parent.obj.par.Sheenroughness.eval() )
	materialParams.Set_transmission( parent.obj.par.Refractiontransmission.eval() )
	materialParams.Set_ior( parent.obj.par.Refractionior.eval() )

	# depending on shading model, we derive the thickness value from one of two parameters.
	# this makes it more intuitive for the user to set the thickness value. from the parameter page that makes sense.
	if parent.obj.par.Shadingmodel.eval() == 'lit':
		materialParams.Set_thickness( parent.obj.par.Refractionthicknessfactor.eval() )
	elif parent.obj.par.Shadingmodel.eval() == 'subsurface':
		materialParams.Set_thickness( parent.obj.par.Subsurfacethicknessuniform.eval() )

	materialParams.Set_absorption(
		parent.obj.par.Refractiontransmittancer.eval(),
		parent.obj.par.Refractiontransmittanceg.eval(),
		parent.obj.par.Refractiontransmittanceb.eval(),
		parent.obj.par.Transmittancedistance.eval()
	)

	materialParams.Set_subsurfaceColor(
		parent.obj.par.Subsurfacecoloruniformr.eval(),
		parent.obj.par.Subsurfacecoloruniformg.eval(),
		parent.obj.par.Subsurfacecoloruniformb.eval()
	)

	materialParams.Set_subsurfacePower( parent.obj.par.Subsurfacepoweruniform.eval() )
	values = materialParams.To_List()

	scriptOp.numSamples = len(values)
	c = scriptOp.appendChan('materialParams')
	c.vals = values

	return
