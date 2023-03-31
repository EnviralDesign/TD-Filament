

/////////////////////////////////////////////////////
////////// VERTEX SHADER ONLY FUNCTIONS /////////////
/////////////////////////////////////////////////////
#if defined(TD_VERTEX_SHADER)

in int td_instancing_enabled;

#endif


/////////////////////////////////////////////////////
///////// FRAGMENT SHADER ONLY FUNCTIONS ////////////
/////////////////////////////////////////////////////
#if defined(TD_PIXEL_SHADER)

int TDInstanceID() {
    return instance_index;
}

// we only want to define this in a pixel shader, as vertex has no use for it and neither does a glsl TOP.
#ifndef IS_POSTPROCESS
    #define uTDPass 0
#endif

// cheap hack fix for now for the fragment shader, we really need to pass this through from the vertex shader.
#ifndef td_instancing_enabled
    #define td_instancing_enabled 0
#endif

vec3 TD_Inverse_Tonemap_Filmic(vec3 x) {
return (0.03 - 0.59 * x - sqrt(0.0009 + 1.3702 * x - 1.0127 * x * x)) / (-5.02 + 4.86 * x);
}

vec3 TD_inverseTonemapSRGB(vec3 color) {
color = clamp(color, 0.0, 1.0);
return TD_Inverse_Tonemap_Filmic(pow(color, vec3(2.2)));
}

#if defined(IS_SHADING_MODEL)

vec2 tdGetUV0() {
    return vec2( vertex_uv01.xy );
}

uint TDCameraIndex()
{
    /*
    if in frag shader, there isn't a tdCameraIndex() function, so we define it using the cam index from vertex stage.
    */
    return vertex_camera_index;
}

// if the first cube map mip level is defined, save to assume the rest are too.
// #ifdef CUBE_SAMPLERS_EXIST

ivec2 IblCubeMip_TextureSize( samplerCube nullFilamentSampler , int mipLevel ){
    
    ivec2 texsize = ivec2(0);

    if( mipLevel == 0 ){
        texsize = textureSize( light_iblMip0 , 0 );
    }

    else if( mipLevel == 1 ){
        texsize = textureSize( light_iblMip1 , 0 );
    }

    else if( mipLevel == 2 ){
        texsize = textureSize( light_iblMip2 , 0 );
    }

    else if( mipLevel == 3 ){
        texsize = textureSize( light_iblMip3 , 0 );
    }

    else if( mipLevel == 4 ){
        texsize = textureSize( light_iblMip4 , 0 );
    }

    else if( mipLevel == 5 ){
        texsize = textureSize( light_iblMip5 , 0 );
    }

    else if( mipLevel == 6 ){
        texsize = textureSize( light_iblMip6 , 0 );
    }

    return texsize;
}

vec4 IblCubeMip_TextureLod( samplerCube nullFilamentSampler , vec3 coords , float mipLevel ){
    /* a wrapper function used to sampler a LOD from the multi sampler cube map system. 
    */
    vec4 a = vec4(0);
    vec4 b = vec4(0);

    float mixer = fract(mipLevel);
    int baseMipLevel = int(mipLevel);

    if( baseMipLevel == 0 ){
        a = texture( light_iblMip0 , coords );
        b = texture( light_iblMip1 , coords );
    }

    if( baseMipLevel == 1 ){
        a = texture( light_iblMip1 , coords );
        b = texture( light_iblMip2 , coords );
    }

    if( baseMipLevel == 2 ){
        a = texture( light_iblMip2 , coords );
        b = texture( light_iblMip3 , coords );
    }

    if( baseMipLevel == 3 ){
        a = texture( light_iblMip3 , coords );
        b = texture( light_iblMip4 , coords );
    }

    if( baseMipLevel == 4 ){
        a = texture( light_iblMip4 , coords );
        b = texture( light_iblMip5 , coords );
    }

    if( baseMipLevel == 5 ){
        a = texture( light_iblMip5 , coords );
        b = texture( light_iblMip6 , coords );
    }

    return mix( a , b , mixer );

}

// seems there's no place in the fragment shader that cares about this like the IBL variant above, but keeping for a while.
// ivec3 ssr_TextureSize( sampler2DArray nullFilamentSampler , int mipLevel ){
    
//     ivec3 texsize = ivec3(0);

//     if( mipLevel == 0 ){
//         texsize = textureSize( light_ssrMip0 , 0 );
//     }

//     else if( mipLevel == 1 ){
//         texsize = textureSize( light_ssrMip1 , 0 );
//     }

//     else if( mipLevel == 2 ){
//         texsize = textureSize( light_ssrMip2 , 0 );
//     }

//     else if( mipLevel == 3 ){
//         texsize = textureSize( light_ssrMip3 , 0 );
//     }

//     else if( mipLevel == 4 ){
//         texsize = textureSize( light_ssrMip4 , 0 );
//     }

//     else if( mipLevel == 5 ){
//         texsize = textureSize( light_ssrMip5 , 0 );
//     }

//     return texsize;
// }

vec4 ssr_textureLod( sampler2DArray nullFilamentSampler , vec3 coords , float mipLevel ){

    vec4 a = vec4(0);
    vec4 b = vec4(0);

    float mixer = fract(mipLevel);
    int baseMipLevel = int(mipLevel);

    if( baseMipLevel == 0 ){
        a = texture( light_ssrMip0 , coords.xy );
        b = texture( light_ssrMip1 , coords.xy );
    }

    if( baseMipLevel == 1 ){
        a = texture( light_ssrMip1 , coords.xy );
        b = texture( light_ssrMip2 , coords.xy );
    }

    if( baseMipLevel == 2 ){
        a = texture( light_ssrMip2 , coords.xy );
        b = texture( light_ssrMip3 , coords.xy );
    }

    if( baseMipLevel == 3 ){
        a = texture( light_ssrMip3 , coords.xy );
        b = texture( light_ssrMip4 , coords.xy );
    }

    if( baseMipLevel == 4 ){
        a = texture( light_ssrMip4 , coords.xy );
        b = texture( light_ssrMip5 , coords.xy );
    }

    if( baseMipLevel == 5 ){
        a = texture( light_ssrMip5 , coords.xy );
        b = texture( light_ssrMip6 , coords.xy );
    }

    return mix( a , b , mixer );

}

mat3 rotationMatrix3Dy(float angle) {
	float s = sin(angle);
	float c = cos(angle);

	return mat3(
		c, 0.0, -s,
		0.0, 1.0, 0.0,
		s, 0.0, c
	);
}

#endif

#endif


/////////////////////////////////////////////////////
////////// PER RENDERABLE OBJECT UNIFORMS ///////////
/////////////////////////////////////////////////////

// NOTE: we are taking in the int i variable, but it's likely we will not even use this in the interest of using TD's instance index instead.

#if defined(IS_SHADING_MODEL)

mat4 objectUniforms_worldFromModelMatrix(int i)
{
    if(td_instancing_enabled == 0){
        return uTDMats[TDCameraIndex()].world;
    }
    else{
        return uTDMats[TDCameraIndex()].world * TDInstanceMat(i);
    }
}

mat3 objectUniforms_worldFromModelNormalMatrix(int i)
{
    if(td_instancing_enabled == 0){
        return uTDMats[TDCameraIndex()].worldForNormals;
    }
    else{
        mat3 inverseTranspose = mat3(transpose(inverse(TDInstanceMat(i))));
        return uTDMats[TDCameraIndex()].worldForNormals * inverseTranspose;
    }
}

uint objectUniforms_morphTargetCount(int i)
{
    // TODO
    // need to derive this from TD cpu land eventually.
    return 0;
}

uint objectUniforms_flagsChannels(int i)
{
    // TODO
    /* 
    for now doing the entire flags packing in here for VISIBILITY, but this obviously needs to be done in CPU land later.
    for performance reasons but also that's where the information needs to come from.

    reference for what the pack Flags Channels function could look like in glsl, we're just going to do it inline though.

    uint packFlagsChannels(
        bool skinning, bool morphing, bool contactShadows, uint channels) {
        return (skinning       ? 0x100u : 0u) |
            (morphing       ? 0x200u : 0u) |
            (contactShadows ? 0x400u : 0u) |
            channels;
    }
    */

    bool skinning = false;
    bool morphing = false;
    bool contactShadows = false;
    uint channels = 0xff; //  0xff is equal to 1111 1111. 8 channels.

    uint packed = 
    (skinning       ? 0x100u : 0u) |
    (morphing       ? 0x200u : 0u) |
    (contactShadows ? 0x400u : 0u) |
    channels;

    return packed;
}

uint objectUniforms_objectId(int i)
{
    // TODO
    // not really used anywhere in the shader, so leaving at -1 for now until we find a use for it.
    return -1;
}

float objectUniforms_userData(int i)
{
    // TODO
    // not really used anywhere in the shader, so leaving at -1 for now until we find a use for it.
    return -1;
}

#endif

/////////////////////////////////////////////////////
////////// FRAME UNIFORMS ///////////////////////////
/////////////////////////////////////////////////////

mat4 frameUniforms_viewFromWorldMatrix()
{
    // cameras view matrix, takes point in world space, puts it in view space.
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].cam;
    #endif
    return mat4(1);
    
}

mat4 frameUniforms_worldFromViewMatrix()
{
    // cameras inverse view matrix, takes point in view space, puts it in world space.
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].camInverse;
    #endif
    return mat4(1);
    
}

mat4 frameUniforms_clipFromViewMatrix()
{
    // cameras projection matrix, takes point in view space, puts it in clip space.
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].proj;
    #endif
    return mat4(1);
    
}

mat4 frameUniforms_viewFromClipMatrix()
{
    // cameras inverse projection matrix, takes point in clip space, puts it in view space.
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].projInverse;
    #endif
    return mat4(1);
    
}

mat4 frameUniforms_clipFromWorldMatrix()
{
    // takes point in world space, puts it in clip space.
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].camProj;
    #endif
    return mat4(1);
    
}

mat4 frameUniforms_worldFromClipMatrix()
{
    // takes point in clip space, puts it in world space.
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].camProjInverse;
    #endif
    return mat4(1);
    
}

float frameUniforms_time()
{
    return uFrameUniforms[0];
}

vec4 frameUniforms_userTime()
{
    return vec4(uFrameUniforms[1], uFrameUniforms[2], uFrameUniforms[3], uFrameUniforms[4]);
}

vec4 frameUniforms_resolution()
{
    // td prepares this information exactly the same way filament does, yay!
    #if defined(IS_SHADING_MODEL)
    return uTDGeneral.viewport;
    #else
    return uTDOutputInfo.res;
    #endif
}

vec3 frameUniforms_worldOffset()
{
    // TODO
    // we dont need to worry about setting this up yet, this is an optimziation to keep higher precision closer to the camera, but saving for later.
    return vec3(uFrameUniforms[5], uFrameUniforms[6], uFrameUniforms[7]);
}

vec3 frameUniforms_cameraPosition()
{
    /*
    The position of the camera in a scene can be obtained by accessing the translation component of the view matrix. 
    The translation component of the view matrix is stored in the last column of the matrix.
    */
    #if defined(IS_SHADING_MODEL)
    return uTDMats[TDCameraIndex()].camInverse[3].xyz;
    #endif
    return vec3(0);
}

float frameUniforms_ev100()
{
    return uFrameUniforms[8];
}

float frameUniforms_exposure()
{
    return uFrameUniforms[9];
}

vec2 frameUniforms_clipControl()
{

    return vec2(uFrameUniforms[10] , uFrameUniforms[11]);
}

vec2 frameUniforms_logicalViewportScale()
{
    return vec2(uFrameUniforms[12] , uFrameUniforms[13]);
}

vec2 frameUniforms_logicalViewportOffset()
{
    return vec2(uFrameUniforms[14] , uFrameUniforms[15]);
}

vec4 frameUniforms_cascadeSplits()
{
    // TODO
    // going to be fun getting this to work later...
    return vec4(
        uFrameUniforms[16], 
        uFrameUniforms[17],
        uFrameUniforms[18],
        uFrameUniforms[19]
        );
}

uint frameUniforms_cascades()
{
    return uint(uFrameUniforms[20]);
}

float frameUniforms_fogDensity()
{
    return uFrameUniforms[21];
}

float frameUniforms_fogHeightFalloff()
{
    return uFrameUniforms[22];
}

float frameUniforms_fogStart()
{
    return uFrameUniforms[23];
}

float frameUniforms_fogMaxOpacity()
{
    return uFrameUniforms[24];
}

vec3 frameUniforms_fogColor()
{
    return vec3(
        uFrameUniforms[25], 
        uFrameUniforms[26],
        uFrameUniforms[27]
    );
}

float frameUniforms_fogColorFromIbl()
{
    return uFrameUniforms[28];
}

float frameUniforms_iblRoughnessOneLevel()
{
    return uFrameUniforms[29];
}

float frameUniforms_iblLuminance()
{
    return uFrameUniforms[30];
}

float frameUniforms_fogInscatteringSize()
{
    return uFrameUniforms[31];
}

float frameUniforms_fogInscatteringStart()
{
    return uFrameUniforms[32];
}

vec4 frameUniforms_lightColorIntensity()
{
    return vec4(uFrameUniforms[33], uFrameUniforms[34], uFrameUniforms[35], uFrameUniforms[36]);
}

vec3 frameUniforms_lightDirection()
{
    return vec3( uFrameUniforms[37], uFrameUniforms[38], uFrameUniforms[39] );
}

float frameUniforms_temporalNoise()
{
    return uFrameUniforms[40];
}

float frameUniforms_shadowPenumbraRatioScale()
{
    return uFrameUniforms[41];
}

uint frameUniforms_directionalShadows()
{
    return uint(uFrameUniforms[42]);
}

float frameUniforms_ssContactShadowDistance()
{
    return uFrameUniforms[43];
}

float frameUniforms_vsmDepthScale()
{
    return uFrameUniforms[44];
}

float frameUniforms_vsmLightBleedReduction()
{
    return uFrameUniforms[45];
}

float frameUniforms_vsmExponent()
{
    return uFrameUniforms[46];
}

uint frameUniforms_shadowSamplingType()
{
    return uint(uFrameUniforms[47]);
} 

float frameUniforms_aoSamplingQualityAndEdgeDistance()
{
    return uFrameUniforms[48];
}

float frameUniforms_cameraFar()
{
    return uFrameUniforms[49];
}

float frameUniforms_aoBentNormals()
{
    return uFrameUniforms[50];
}

vec3[9] frameUniforms_iblSH(){
    
    vec3[9] sh;
    sh[0] = vec3(uFrameUniforms[51], uFrameUniforms[52], uFrameUniforms[53]);
    sh[1] = vec3(uFrameUniforms[54], uFrameUniforms[55], uFrameUniforms[56]);
    sh[2] = vec3(uFrameUniforms[57], uFrameUniforms[58], uFrameUniforms[59]);
    sh[3] = vec3(uFrameUniforms[60], uFrameUniforms[61], uFrameUniforms[62]);
    sh[4] = vec3(uFrameUniforms[63], uFrameUniforms[64], uFrameUniforms[65]);
    sh[5] = vec3(uFrameUniforms[66], uFrameUniforms[67], uFrameUniforms[68]);
    sh[6] = vec3(uFrameUniforms[69], uFrameUniforms[70], uFrameUniforms[71]);
    sh[7] = vec3(uFrameUniforms[72], uFrameUniforms[73], uFrameUniforms[74]);
    sh[8] = vec3(uFrameUniforms[75], uFrameUniforms[76], uFrameUniforms[77]);

    return sh;

}

uint frameUniforms_lightChannels()
{
    return uint(uFrameUniforms[78]);
}

vec2 frameUniforms_froxelCountXY()
{
    return vec2(uFrameUniforms[79], uFrameUniforms[80]);
}

vec4 frameUniforms_zParams()
{
    return vec4(uFrameUniforms[81], uFrameUniforms[82], uFrameUniforms[83], uFrameUniforms[84]);
}

uvec3 frameUniforms_fParams()
{
    return uvec3(uFrameUniforms[85], uFrameUniforms[86], uFrameUniforms[87]);
}

vec2 frameUniforms_lightFarAttenuationParams()
{
    return vec2(uFrameUniforms[88], uFrameUniforms[89]);
}

// lodBias, float
float frameUniforms_lodBias()
{
    return uFrameUniforms[90];
}

float frameUniforms_refractionLodOffset()
{
    return uFrameUniforms[91];
}

////////////////////////////////////////////////////////////////////////////////////////
// Material Uniforms ///////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

#if defined(TD_PIXEL_SHADER)

vec3 materialParams_baseColor()
{
    // init
    vec3 baseColor = vec3(1.0);

    #ifdef BASE_COLOR_METHOD_0 // uniform only
        baseColor = vec3( uMaterialParams[0][uTDPass], uMaterialParams[1][uTDPass], uMaterialParams[2][uTDPass] );
    #endif

    #ifdef BASE_COLOR_METHOD_1 // top only
        baseColor = TD_inverseTonemapSRGB( texture( mat_baseColor , tdGetUV0() ).rgb );
    #endif

    #ifdef BASE_COLOR_METHOD_2 // uniform + top
        baseColor = TD_inverseTonemapSRGB( texture( mat_baseColor , tdGetUV0() ).rgb );
        baseColor += vec3( uMaterialParams[0][uTDPass], uMaterialParams[1][uTDPass], uMaterialParams[2][uTDPass] );
    #endif

    #ifdef BASE_COLOR_METHOD_3 // uniform * top
        baseColor = TD_inverseTonemapSRGB( texture( mat_baseColor , tdGetUV0() ).rgb );
        baseColor *= vec3( uMaterialParams[0][uTDPass], uMaterialParams[1][uTDPass], uMaterialParams[2][uTDPass] );
    #endif


    return baseColor;
}

vec3 materialParams_normal()
{
    // we init the normals as if they are 0-1 space.
    vec3 normals = vec3(0.0, 0.0, 1.0);

    #ifdef NORMAL_METHOD_0 // no normals
        // nothing really needs to go here, but place holder for now.
    #endif

    #ifdef NORMAL_METHOD_1 // top only
        float normalStrength = uMaterialParams[18][uTDPass];
        normals = texture( mat_normal , tdGetUV0() ).xyz;
        #ifdef NORMAL_STYLE_1 // if normals are directX style, flip the y before we expand to -1:1 space.
            normals.y = 1 - normals.y;
        #endif
        normals = normals * 2.0 - 1.0;
        normals.xy *= normalStrength;
    #endif

    return normals;
}

vec3 materialParams_bentNormal()
{
    // we init the bent normals as if they are 0-1 space.
    vec3 bentNormals = vec3(0.0, 0.0, 1.0);

    #ifdef BENT_NORMAL_METHOD_0 // no bent normals
        // nothing really needs to go here, but place holder for now.
    #endif

    #ifdef BENT_NORMAL_METHOD_1 // top only
        bentNormals = texture( mat_bentNormal , tdGetUV0() ).xyz;
        #ifdef BENT_NORMAL_STYLE_1 // if bent normals are directX style, flip the y before we expand to -1:1 space.
            bentNormals.y = 1 - bentNormals.y;
        #endif
        bentNormals = bentNormals * 2.0 - 1.0;
    #endif

    return bentNormals;
}

float materialParams_ambientOcclusion()
{
    float ao = 1.0;

    #ifdef AMBIENT_OCCLUSION_METHOD_0 // none
        // nothing really needs to go here, but place holder for now.
    #endif

    #ifdef AMBIENT_OCCLUSION_METHOD_1 // weighted top
        float ao_weight = uMaterialParams[17][uTDPass];
        ao = mix( 1.0 , texture( mat_ambientOcclusion , tdGetUV0() ).r , ao_weight );
    #endif

    return ao;
}

float materialParams_metallic()
{
    float metalness = 0.0;

    #ifdef METALNESS_METHOD_0 // uniform only
        metalness = uMaterialParams[3][uTDPass];
    #endif

    #ifdef METALNESS_METHOD_1 // top only
        metalness = texture( mat_metalness , tdGetUV0() ).r;
    #endif

    return metalness;
}

float materialParams_roughness()
{
    float roughness = 0.4;

    #ifdef ROUGHNESS_METHOD_0 // uniform only
        roughness = uMaterialParams[4][uTDPass];
    #endif

    #ifdef ROUGHNESS_METHOD_1 // top only
        roughness = texture( mat_roughness , tdGetUV0() ).r;
    #endif

    return roughness;
}

float materialParams_reflectance()
{
    float reflectance = 0.0;

    #ifdef REFLECTANCE_METHOD_0 // uniform only
        reflectance = uMaterialParams[5][uTDPass];
    #endif

    #ifdef REFLECTANCE_METHOD_1 // top only
        reflectance = texture( mat_reflectance , tdGetUV0() ).r;
    #endif

    return reflectance;
}

float materialParams_specularAntiAliasingVariance()
{
    return uMaterialParams[34][uTDPass];
}

float materialParams_specularAntiAliasingThreshold()
{
    return uMaterialParams[35][uTDPass];
}

float materialParams_clearCoat()
{
    float clearCoatFactor = 0;

    #if defined(CLEAR_COAT_METHOD_1)
        clearCoatFactor = uMaterialParams[10][uTDPass];
    #endif

    #if defined(CLEAR_COAT_METHOD_2)
        clearCoatFactor = texture( mat_clearCoat , tdGetUV0() ).r;
    #endif

    return clearCoatFactor;
}

float materialParams_clearCoatRoughness()
{
    float clearCoatRoughness = 0.0;

    #if defined(CLEAR_COAT_ROUGHNESS_METHOD_0)
        clearCoatRoughness = uMaterialParams[11][uTDPass];
    #endif

    #if defined(CLEAR_COAT_ROUGHNESS_METHOD_1)
        clearCoatRoughness = texture( mat_clearCoatRoughness , tdGetUV0() ).r;
    #endif

    return clearCoatRoughness;
}

vec3 materialParams_clearCoatNormal()
{
    vec3 clearCoatNormal = vec3(0.0, 0.0, 1.0);

    #if defined(CLEAR_COAT_NORMAL_METHOD_0)
        // nothing really needs to go here, but place holder for now.
    #endif

    #if defined(CLEAR_COAT_NORMAL_METHOD_1)
        clearCoatNormal = texture( mat_clearCoatNormal , tdGetUV0() ).xyz;
        #ifdef CLEAR_COAT_NORMAL_STYLE_1 // if normals are directX style, flip the y before we expand to -1:1 space.
            clearCoatNormal.y = 1 - clearCoatNormal.y;
        #endif
        clearCoatNormal = clearCoatNormal * 2.0 - 1.0;
        clearCoatNormal.xy *= uMaterialParams[12][uTDPass];
        clearCoatNormal = normalize(clearCoatNormal);
    #endif
    return clearCoatNormal;
}

vec3 materialParams_sheenColor()
{
    vec3 sheenColor = vec3(0.0, 0.0, 0.0);

    #if defined(SHEEN_METHOD_0)
        // nothing really needs to go here, but place holder for now.
    #endif

    #if defined(SHADING_MODEL_CLOTH) && defined(SHEEN_METHOD_0)
        sheenColor = vec3( uMaterialParams[6][uTDPass], uMaterialParams[7][uTDPass], uMaterialParams[8][uTDPass] );
    #endif

    #if defined(SHEEN_METHOD_1)
        sheenColor = vec3( uMaterialParams[6][uTDPass], uMaterialParams[7][uTDPass], uMaterialParams[8][uTDPass] );
    #endif

    #if defined(SHEEN_METHOD_2)
        sheenColor = texture( mat_sheenColor , tdGetUV0() ).rgb;
    #endif

    return sheenColor;
}

float materialParams_sheenRoughness()
{
    float sheenRoughness = 0.0;

    #if defined(SHEEN_ROUGHNESS_METHOD_0)
        sheenRoughness = uMaterialParams[9][uTDPass];
    #endif

    #if defined(SHEEN_ROUGHNESS_METHOD_1)
        sheenRoughness = texture( mat_sheenRoughness , tdGetUV0() ).r;
    #endif

    return sheenRoughness;
}

float materialParams_anisotropy()
{
    float anisotropy = 0.0;

    #if defined(ANISOTROPY_METHOD_0)
        // nothing really needs to go here, but place holder for now.
    #endif

    #if defined(ANISOTROPY_METHOD_1)
        anisotropy = uMaterialParams[13][uTDPass] * 2 - 1;
        
    #endif

    #if defined(ANISOTROPY_METHOD_2)
        anisotropy = texture( mat_anisotropy , tdGetUV0() ).r * 2 - 1;
    #endif

    return anisotropy;
}

vec3 materialParams_anisotropyDirection()
{
    vec3 anisotropyDirection = vec3(1.0, 0.0, 0.0);

    #if defined(ANISOTROPY_DIRECTION_METHOD_0)
        anisotropyDirection = vec3( uMaterialParams[14][uTDPass], uMaterialParams[15][uTDPass], uMaterialParams[16][uTDPass] ) * 2 - 1;
    #endif

    #if defined(ANISOTROPY_DIRECTION_METHOD_1)
        anisotropyDirection = texture( mat_anisotropyDirection , tdGetUV0() ).xyz * 2 - 1;
        anisotropyDirection = normalize(anisotropyDirection);
    #endif

    return anisotropyDirection;
}

float materialParams_transmission()
{
    return uMaterialParams[28][uTDPass];
}

float materialParams_ior()
{
    
    return uMaterialParams[27][uTDPass];
}

float materialParams_thickness()
{
    float thickness = 1.0;

    #if defined(MATERIAL_HAS_REFRACTION)

        #if defined(THICKNESS_SOURCE_0)
            thickness = uMaterialParams[32][uTDPass];
        #endif

        #if defined(THICKNESS_SOURCE_1)
            thickness = texture( mat_thickness , tdGetUV0() ).r * uMaterialParams[32][uTDPass];
        #endif

    #endif

    #if defined(SHADING_MODEL_SUBSURFACE)
        #if defined(SUBSURFACE_THICKNESS_SOURCE_0)
            thickness = uMaterialParams[32][uTDPass];
        #endif
        #if defined(SUBSURFACE_THICKNESS_SOURCE_1)
            thickness = texture( mat_subsurfaceThickness , tdGetUV0() ).r * uMaterialParams[32][uTDPass];
        #endif
    #endif

    return thickness;
}

float materialParams_microThickness()
{
    return materialParams_thickness();
}

vec3 materialParams_absorption()
{
    vec3 absorption = vec3(
        uMaterialParams[29][uTDPass],
        uMaterialParams[30][uTDPass],
        uMaterialParams[31][uTDPass]
    );
    return absorption;
}

vec4 materialParams_emissive()
{
    vec4 emissive = vec4(0.0, 0.0, 0.0, 0.0);

    #if defined(EMISSIVE_METHOD_0)
        // nothing really needs to go here, but place holder for now.
    #endif

    #if defined(EMISSIVE_METHOD_1)
        emissive = vec4(
            uMaterialParams[19][uTDPass],
            uMaterialParams[20][uTDPass],
            uMaterialParams[21][uTDPass],
            uMaterialParams[22][uTDPass]
        );
    #endif

    #if defined(EMISSIVE_METHOD_2)
        emissive.rgb = TD_inverseTonemapSRGB(texture( mat_emissive , tdGetUV0() ).rgb) * uMaterialParams[47][uTDPass];
        emissive.a = uMaterialParams[22][uTDPass]; // always get the exposure weight from parameter for now
    #endif
    return emissive;
}

vec4 materialParams_postLightingColor()
{
    vec4 postLightingColor = vec4(0.0, 0.0, 0.0, 0.0);

    #if defined(POST_LIGHTING_METHOD_0)
        // nothing really needs to go here, but place holder for now.
    #endif

    #if defined(POST_LIGHTING_METHOD_1)
        postLightingColor = vec4(
            uMaterialParams[23][uTDPass],
            uMaterialParams[24][uTDPass],
            uMaterialParams[25][uTDPass],
            uMaterialParams[26][uTDPass]
        );
    #endif

    #if defined(POST_LIGHTING_METHOD_2)
        vec4 postlight_srgb = texture( mat_postLightingColor , tdGetUV0() );
        postLightingColor.rgb = TD_inverseTonemapSRGB(postlight_srgb.rgb);
        postLightingColor.a = postlight_srgb.a;
    #endif

    return postLightingColor;
}

vec3 materialParams_subsurfaceColor()
{
    vec3 subsurfaceColor = vec3(0);

    #if defined(SUBSURFACE_COLOR_METHOD_0)
        subsurfaceColor = vec3(
            uMaterialParams[34][uTDPass],
            uMaterialParams[35][uTDPass],
            uMaterialParams[36][uTDPass]
        );
    
    #endif

    #if defined(SUBSURFACE_COLOR_METHOD_1)
        subsurfaceColor = texture( mat_subsurfaceColor , tdGetUV0() ).rgb;
    #endif

    return subsurfaceColor;
}

float materialParams_subsurfacePower()
{
    float subsurfacePower = 0.0;

    #if defined(SUBSURFACE_POWER_METHOD_0)
        subsurfacePower = uMaterialParams[33][uTDPass];
    #endif

    #if defined(SUBSURFACE_POWER_METHOD_1)
        subsurfacePower = texture( mat_subsurfacePower , tdGetUV0() ).r * uMaterialParams[33][uTDPass]; 
    #endif

    return subsurfacePower;
}

float materialParams_maskThreshold()
{
    return uMaterialParams[37][uTDPass];
}

bool materialParams_doubleSided()
{
    return uMaterialParams[40][uTDPass] > 0.0;
}

float materialParams_alpha()
{
    return uMaterialParams[48][uTDPass];
}

///////////////////// these items are for Gaussian blur mip mapping /////////////////////////////
vec2 materialParams_axis()
{
    vec2 axis = vec2(
        uMaterialParams[41][uTDPass],
        uMaterialParams[42][uTDPass]
    );
    return axis;
}

float materialParams_level()
{
    return uMaterialParams[43][uTDPass];
}

float materialParams_layer()
{
    return uMaterialParams[44][uTDPass];
}

int materialParams_reinhard()
{
    return int(uMaterialParams[45][uTDPass]);
}

int materialParams_count()
{
    return int(uMaterialParams[46][uTDPass]);
}

vec2 materialParams_kernel(uint i)
{
    return vec2( uMaterialParams[128 + (i * 2)][uTDPass] , uMaterialParams[129 + (i * 2 + 1)][uTDPass] );
}


#endif

////////////////////////////////////////////////////////////////////////////////////////
// Shadow Uniforms /////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

float shadowUniforms_bulbRadiusLs(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.
    return 0.02;
}

float shadowUniforms_nearOverFarMinusNear(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.

    float n = 0.1;
    float f = 1000.0;

    float nearOverFarMinusNear = n / (f - n);

    return nearOverFarMinusNear;
}

vec4 shadowUniforms_scissorNormalized(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.

    float dim = 1024.0; // mapsize can be anywhere from 8 - 2048
    uint border = 1u;

    float l = border;
    float b = border;
    float w = dim - 2u * border;
    float h = dim - 2u * border;

    float texel = 1.0 / dim;
    
    vec4 v = vec4( l, b, l + w, b + h ) * texel;

    bool textureSpaceFlipped = false;

    if (textureSpaceFlipped) {
        v =  vec4(v.x, 1.0f - v.w, v.z, 1.0f - v.y);
    }

    return v;
}

uint shadowUniforms_layer(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.
    return 0;
}

bool shadowUniforms_elvsm(uint i) 
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.
    return false;
}

float shadowUniforms_normalBias(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.
    return 0.0;
}

mat4 shadowUniforms_lightFromWorldMatrix(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.
    return mat4(1.0);
}

vec4 shadowUniforms_lightFromWorldZ(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple shadow/lights hence the argument.
    return vec4(1.0);
}

////////////////////////////////////////////////////////////////////////////////////////
// Froxel Uniforms /////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

uvec4 froxelRecordUniforms_records(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple froxels hence the argument.
    return uvec4(0, 0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////////////
// Light Uniforms /////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////

vec4 lightsUniforms_positionFalloff(uint i)
{
    
    // TODO
    float lightRadius = 100; // just a made up radius, this will all get calc'ed in the cpu side.
    float sqFalloff = lightRadius * lightRadius;
    float squaredFallOffInv = sqFalloff > 0.0 ? (1 / sqFalloff) : 0.0;

    vec3 lightpos = vec3(0,2,0);

    return vec4(lightpos, squaredFallOffInv);
}

vec3 lightsUniforms_direction(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    // obviously we will need to deal with multiple lights hence the argument.
    return vec3(0,-1,0);
}

vec4 lightsUniforms_colorIES(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.
    
    vec3 lightcolor = vec3(1);
    float IES_profile = 0.0;

    return vec4(lightcolor, IES_profile);
}

vec2 lightsUniforms_scaleOffset(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.

    // some dummy data. this entire function's contents pretty much would get calc'ed on the cpu side..
    float inner = 35;
    float outer = 45;

    // glsl version of above:
    float DEG_TO_RAD = 0.01745329251994329576923690768489;
    float PI_2 = 1.5707963267948966192313216916398;
    float innerClamped = clamp(abs(inner), 0.5f * DEG_TO_RAD, PI_2);
    float outerClamped = clamp(abs(outer), 0.5f * DEG_TO_RAD, PI_2);

    // inner must always be smaller than outer
    innerClamped = min(innerClamped, outerClamped);

    float cosOuter = cos(outerClamped);
    float cosInner = cos(innerClamped);

    float scale = 1.0f / max(1.0f / 1024.0, cosInner - cosOuter);
    float offset = -cosOuter * scale;

    vec2 scaleOffset = vec2( scale, offset );

    return scaleOffset;
}


float lightsUniforms_intensity(uint i)
{
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.

    const float ONE_OVER_PI_ = 0.318309886183790671537767526745028724;
    const float PI_ = 3.1415926535897932384626433832795;
    const float TAU_ = 6.283185307179586476925286766559;

    // unit type, 0 == lux, 1 == candela
    uint unitType = 0;

    // Sun/Directional == 0, point == 1, focused spot == 2, spot == 3
    uint lightType = 1;
    float intensity = 1000.0;

    float luminousPower = intensity;
    float luminousIntensity = 0;

    if (lightType == 0) { // sun/directional
        luminousIntensity = luminousPower; // luminousPower in lux, nothing to do.
    }

    if (lightType == 1) { // 
        
        if(unitType == 0) { // lux
            luminousIntensity = luminousPower * ONE_OVER_PI_ * 0.25;
        } else { // candela
            luminousIntensity = luminousPower;
        }

    }

    if (lightType == 2) { // focused spot

        float cosOuter = 0.125;// comes from the setIntensity() function in c++

        if(unitType == 0) { // lux
            
            luminousIntensity = luminousPower / (TAU_ * (1.0f - cosOuter));
        } else { // candela

                luminousIntensity = luminousPower;
                
                // lp = li * (2 * pi * (1 - cos(cone_outer / 2)))
                luminousPower = luminousIntensity * (TAU_ * (1.0 - cosOuter));
        }

    }

    if (lightType == 3) { // 

        if(unitType == 0) { // lux
            
            luminousIntensity = luminousPower * ONE_OVER_PI_;
            
        } else { // candela

                luminousIntensity = luminousPower;
        }

    }

    return luminousIntensity;
}


uint lightsUniforms_typeShadow(uint i)
{
    // TODO
    bool isPointLight = false;
    
    uint type = isPointLight ? 0u : 1u;
    bool contactShadow = false;
    uint index = 0;

    return (type & 0xF) | (contactShadow ? 0x10 : 0x00) | (index << 8);

}

//channels, uint
uint lightsUniforms_channels(uint i)
{
    
    // TODO
    // sensible default in filament, we'll need to hook this up to td later.

    uint lightChannels = 0xff; //  0xff is equal to 1111 1111. 8 channels.
    bool castShadows = false;

    return lightChannels | (castShadows ? 0x10000 : 0);
}

/////////////////////////////////////////////////////
////////// BONES UNIFORMS ///////////////////////////
/////////////////////////////////////////////////////

uvec4 boneUniforms_cof(uint i)
{
    // TODO
    // we'll need to hook this up to td later.
    return uvec4(1.0);
}

mat3x4 boneUniforms_transform(uint i)
{
    // TODO
    // we'll need to hook this up to td later.
    return mat3x4(1.0);
}

/////////////////////////////////////////////////////
////////// MORPHGING UNIFORMS ///////////////////////
/////////////////////////////////////////////////////

vec4 morphingUniforms_weights(uint i)
{
    // TODO
    // we'll need to hook this up to td later.
    return vec4(1.0);
}

/////////////////////////////////////////////////////
//////////// CUSTOM UNIFORMS ////////////////////////
/////////////////////////////////////////////////////

float customUniforms_iblRotation()
{
    return uCustomUniforms[0];
}