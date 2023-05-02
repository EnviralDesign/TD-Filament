#version 410 core

#extension GL_ARB_shading_language_packing : enable

#extension GL_GOOGLE_cpp_style_line_directive : enable

#define TARGET_GL_ENVIRONMENT

#define FILAMENT_OPENGL_SEMANTICS

#define FILAMENT_HAS_FEATURE_TEXTURE_GATHER
precision highp float;
precision highp int;

#ifndef SPIRV_CROSS_CONSTANT_ID_0
#define SPIRV_CROSS_CONSTANT_ID_0 1
#endif
const int BACKEND_FEATURE_LEVEL = SPIRV_CROSS_CONSTANT_ID_0;

#ifndef SPIRV_CROSS_CONSTANT_ID_1
#define SPIRV_CROSS_CONSTANT_ID_1 64
#endif
const int CONFIG_MAX_INSTANCES = SPIRV_CROSS_CONSTANT_ID_1;


#if defined(FILAMENT_VULKAN_SEMANTICS)
#define LAYOUT_LOCATION(x) layout(location = x)
#else
#define LAYOUT_LOCATION(x)
#endif
#define bool2    bvec2
#define bool3    bvec3
#define bool4    bvec4
#define int2     ivec2
#define int3     ivec3
#define int4     ivec4
#define uint2    uvec2
#define uint3    uvec3
#define uint4    uvec4
#define float2   vec2
#define float3   vec3
#define float4   vec4
#define float3x3 mat3
#define float4x4 mat4


struct ShadowData {
highp mat4 lightFromWorldMatrix;
highp vec4 lightFromWorldZ;
highp vec4 scissorNormalized;
mediump float texelSizeAtOneMeter;
mediump float bulbRadiusLs;
mediump float nearOverFarMinusNear;
mediump float normalBias;
bool elvsm;
mediump uint layer;
mediump uint reserved1;
mediump uint reserved2;
};
struct BoneData {
highp mat3x4 transform;
highp uvec4 cof;
};
struct PerRenderableData {
highp mat4 worldFromModelMatrix;
highp mat3 worldFromModelNormalMatrix;
highp uint morphTargetCount;
highp uint flagsChannels;
highp uint objectId;
highp float userData;
highp vec4 reserved[8];
};

#define FILAMENT_QUALITY_LOW    0
#define FILAMENT_QUALITY_NORMAL 1
#define FILAMENT_QUALITY_HIGH   2
#define FILAMENT_QUALITY FILAMENT_QUALITY_HIGH
#define CLEAR_COAT_IOR_CHANGE
#define SPECULAR_AMBIENT_OCCLUSION 1
#define MULTI_BOUNCE_AMBIENT_OCCLUSION 1
#define BLEND_MODE_OPAQUE
#define POST_LIGHTING_BLEND_MODE_TRANSPARENT
#define SHADING_MODEL_UNLIT
#define MATERIAL_HAS_BASE_COLOR
#define SHADING_INTERPOLATION 
#define VARYING in

LAYOUT_LOCATION(4) VARYING highp vec4 vertex_worldPosition;
#if defined(HAS_ATTRIBUTE_TANGENTS)
LAYOUT_LOCATION(5) SHADING_INTERPOLATION VARYING mediump vec3 vertex_worldNormal;
#if defined(MATERIAL_NEEDS_TBN)
LAYOUT_LOCATION(6) SHADING_INTERPOLATION VARYING mediump vec4 vertex_worldTangent;
#endif
#endif
LAYOUT_LOCATION(7) VARYING highp vec4 vertex_position;
LAYOUT_LOCATION(8) flat VARYING highp int instance_index;
#if defined(HAS_ATTRIBUTE_COLOR)
LAYOUT_LOCATION(9) VARYING mediump vec4 vertex_color;
#endif
#if defined(HAS_ATTRIBUTE_UV0) && !defined(HAS_ATTRIBUTE_UV1)
LAYOUT_LOCATION(10) VARYING highp vec2 vertex_uv01;
#elif defined(HAS_ATTRIBUTE_UV1)
LAYOUT_LOCATION(10) VARYING highp vec4 vertex_uv01;
#endif
#if defined(VARIANT_HAS_SHADOWING) && defined(VARIANT_HAS_DIRECTIONAL_LIGHTING)
LAYOUT_LOCATION(11) VARYING highp vec4 vertex_lightSpacePosition;
#endif

LAYOUT_LOCATION(0) in highp vec4 variable_eyeDirection;

layout(std140) uniform FrameUniforms {
    mat4 viewFromWorldMatrix;
    mat4 worldFromViewMatrix;
    mat4 clipFromViewMatrix;
    mat4 viewFromClipMatrix;
    mat4 clipFromWorldMatrix;
    mat4 worldFromClipMatrix;
    vec4 clipTransform;
    lowp vec2 clipControl;
    float time;
    float temporalNoise;
    vec4 userTime;
    vec4 resolution;
    vec2 logicalViewportScale;
    vec2 logicalViewportOffset;
    lowp float lodBias;
    lowp float refractionLodOffset;
    lowp float padding1;
    lowp float padding2;
    vec3 cameraPosition;
    float oneOverFarMinusNear;
    lowp vec3 worldOffset;
    float nearOverFarMinusNear;
    lowp float cameraFar;
    float exposure;
    lowp float ev100;
    lowp float needsAlphaChannel;
    lowp float aoSamplingQualityAndEdgeDistance;
    lowp float aoBentNormals;
    lowp float aoReserved0;
    lowp float aoReserved1;
    lowp vec4 zParams;
    lowp uvec3 fParams;
    lowp uint lightChannels;
    lowp vec2 froxelCountXY;
    lowp float iblLuminance;
    lowp float iblRoughnessOneLevel;
    lowp vec3 iblSH[9];
    lowp vec3 lightDirection;
    lowp float padding0;
    lowp vec4 lightColorIntensity;
    lowp vec4 sun;
    lowp vec2 lightFarAttenuationParams;
    lowp uint directionalShadows;
    lowp float ssContactShadowDistance;
    vec4 cascadeSplits;
    lowp uint cascades;
    lowp float reserved0;
    lowp float reserved1;
    lowp float shadowPenumbraRatioScale;
    lowp float vsmExponent;
    lowp float vsmDepthScale;
    lowp float vsmLightBleedReduction;
    lowp uint shadowSamplingType;
    lowp float fogStart;
    lowp float fogMaxOpacity;
    lowp float fogHeight;
    lowp float fogHeightFalloff;
    lowp vec3 fogColor;
    lowp float fogDensity;
    lowp float fogInscatteringStart;
    lowp float fogInscatteringSize;
    lowp float fogColorFromIbl;
    lowp float fogReserved0;
    mat4 ssrReprojection;
    mat4 ssrUvFromViewMatrix;
    lowp float ssrThickness;
    lowp float ssrBias;
    lowp float ssrDistance;
    lowp float ssrStride;
    lowp vec4 reserved[63];
} frameUniforms;

layout(std140) uniform ObjectUniforms {
    PerRenderableData data[CONFIG_MAX_INSTANCES];
} objectUniforms;

layout(std140) uniform LightsUniforms {
    mat4 lights[256];
} lightsUniforms;

layout(std140) uniform FroxelRecordUniforms {
    uvec4 records[1024];
} froxelRecordUniforms;

layout(std140) uniform MaterialParams {
    int showSun;
    int constantColor;
    vec4 color;
} materialParams;

uniform mediump sampler2DArrayShadow light_shadowMap;
uniform mediump usampler2D light_froxels;
uniform mediump sampler2D light_iblDFG;
uniform mediump samplerCube light_iblSpecular;
uniform mediump sampler2DArray light_ssao;
uniform mediump sampler2DArray light_ssr;
uniform highp sampler2D light_structure;

uniform  samplerCube materialParams_skybox;

float filament_lodBias;

#define PI                 3.14159265359
#define HALF_PI            1.570796327
#define MEDIUMP_FLT_MAX    65504.0
#define MEDIUMP_FLT_MIN    0.00006103515625
#ifdef TARGET_MOBILE
#define FLT_EPS            MEDIUMP_FLT_MIN
#define saturateMediump(x) min(x, MEDIUMP_FLT_MAX)
#else
#define FLT_EPS            1e-5
#define saturateMediump(x) x
#endif
#define saturate(x)        clamp(x, 0.0, 1.0)
#define atan2(x, y)        atan(y, x)
float pow5(float x) {
float x2 = x * x;
return x2 * x2 * x;
}
float sq(float x) {
return x * x;
}
float max3(const vec3 v) {
return max(v.x, max(v.y, v.z));
}
float vmax(const vec2 v) {
return max(v.x, v.y);
}
float vmax(const vec3 v) {
return max(v.x, max(v.y, v.z));
}
float vmax(const vec4 v) {
return max(max(v.x, v.y), max(v.y, v.z));
}
float min3(const vec3 v) {
return min(v.x, min(v.y, v.z));
}
float vmin(const vec2 v) {
return min(v.x, v.y);
}
float vmin(const vec3 v) {
return min(v.x, min(v.y, v.z));
}
float vmin(const vec4 v) {
return min(min(v.x, v.y), min(v.y, v.z));
}
float acosFast(float x) {
float y = abs(x);
float p = -0.1565827 * y + 1.570796;
p *= sqrt(1.0 - y);
return x >= 0.0 ? p : PI - p;
}
float acosFastPositive(float x) {
float p = -0.1565827 * x + 1.570796;
return p * sqrt(1.0 - x);
}
highp vec4 mulMat4x4Float3(const highp mat4 m, const highp vec3 v) {
return v.x * m[0] + (v.y * m[1] + (v.z * m[2] + m[3]));
}
highp vec3 mulMat3x3Float3(const highp mat4 m, const highp vec3 v) {
return v.x * m[0].xyz + (v.y * m[1].xyz + (v.z * m[2].xyz));
}
void toTangentFrame(const highp vec4 q, out highp vec3 n) {
n = vec3( 0.0,  0.0,  1.0) +
vec3( 2.0, -2.0, -2.0) * q.x * q.zwx +
vec3( 2.0,  2.0, -2.0) * q.y * q.wzy;
}
void toTangentFrame(const highp vec4 q, out highp vec3 n, out highp vec3 t) {
toTangentFrame(q, n);
t = vec3( 1.0,  0.0,  0.0) +
vec3(-2.0,  2.0, -2.0) * q.y * q.yxw +
vec3(-2.0,  2.0,  2.0) * q.z * q.zwx;
}
highp mat3 cofactor(const highp mat3 m) {
highp float a = m[0][0];
highp float b = m[1][0];
highp float c = m[2][0];
highp float d = m[0][1];
highp float e = m[1][1];
highp float f = m[2][1];
highp float g = m[0][2];
highp float h = m[1][2];
highp float i = m[2][2];
highp mat3 cof;
cof[0][0] = e * i - f * h;
cof[0][1] = c * h - b * i;
cof[0][2] = b * f - c * e;
cof[1][0] = f * g - d * i;
cof[1][1] = a * i - c * g;
cof[1][2] = c * d - a * f;
cof[2][0] = d * h - e * g;
cof[2][1] = b * g - a * h;
cof[2][2] = a * e - b * d;
return cof;
}
float interleavedGradientNoise(highp vec2 w) {
const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
return fract(m.z * fract(dot(w, m.xy)));
}

PerRenderableData object_uniforms;
void initObjectUniforms(out PerRenderableData p) {
#if defined(MATERIAL_HAS_INSTANCES)
p = objectUniforms.data[0];
#else
p.worldFromModelMatrix = objectUniforms.data[instance_index].worldFromModelMatrix;
p.worldFromModelNormalMatrix = objectUniforms.data[instance_index].worldFromModelNormalMatrix;
p.morphTargetCount = objectUniforms.data[instance_index].morphTargetCount;
p.flagsChannels = objectUniforms.data[instance_index].flagsChannels;
p.objectId = objectUniforms.data[instance_index].objectId;
p.userData = objectUniforms.data[instance_index].userData;
#endif
}
#if defined(MATERIAL_HAS_INSTANCES)
int getInstanceIndex() {
return instance_index;
}
#endif

#if defined(VARIANT_HAS_SHADOWING)
highp vec4 computeLightSpacePosition(highp vec3 p, const highp vec3 n,
const highp vec3 dir, const float b, const highp mat4 lightFromWorldMatrix) {
#if !defined(VARIANT_HAS_VSM)
highp float cosTheta = saturate(dot(n, dir));
highp float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
p += n * (sinTheta * b);
#endif
return mulMat4x4Float3(lightFromWorldMatrix, p);
}
#endif

highp mat3  shading_tangentToWorld;
highp vec3  shading_position;
vec3  shading_view;
vec3  shading_normal;
vec3  shading_geometricNormal;
vec3  shading_reflected;
float shading_NoV;
#if defined(MATERIAL_HAS_BENT_NORMAL)
vec3  shading_bentNormal;
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT)
vec3  shading_clearCoatNormal;
#endif
highp vec2 shading_normalizedViewportCoord;

float luminance(const vec3 linear) {
return dot(linear, vec3(0.2126, 0.7152, 0.0722));
}
float computePreExposedIntensity(const highp float intensity, const highp float exposure) {
return intensity * exposure;
}
void unpremultiply(inout vec4 color) {
color.rgb /= max(color.a, FLT_EPS);
}
vec3 ycbcrToRgb(float luminance, vec2 cbcr) {
const mat4 ycbcrToRgbTransform = mat4(
1.0000,  1.0000,  1.0000,  0.0000,
0.0000, -0.3441,  1.7720,  0.0000,
1.4020, -0.7141,  0.0000,  0.0000,
-0.7010,  0.5291, -0.8860,  1.0000
);
return (ycbcrToRgbTransform * vec4(luminance, cbcr, 1.0)).rgb;
}
vec3 Inverse_Tonemap_Filmic(const vec3 x) {
return (0.03 - 0.59 * x - sqrt(0.0009 + 1.3702 * x - 1.0127 * x * x)) / (-5.02 + 4.86 * x);
}
vec3 inverseTonemapSRGB(vec3 color) {
color = clamp(color, 0.0, 1.0);
return Inverse_Tonemap_Filmic(pow(color, vec3(2.2)));
}
vec3 inverseTonemap(vec3 linear) {
return Inverse_Tonemap_Filmic(clamp(linear, 0.0, 1.0));
}
vec3 decodeRGBM(vec4 c) {
c.rgb *= (c.a * 16.0);
return c.rgb * c.rgb;
}
highp vec2 getFragCoord(const highp vec2 resolution) {
#if defined(TARGET_METAL_ENVIRONMENT) || defined(TARGET_VULKAN_ENVIRONMENT)
return vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
#else
return gl_FragCoord.xy;
#endif
}
vec3 heatmap(float v) {
vec3 r = v * 2.1 - vec3(1.8, 1.14, 0.3);
return 1.0 - r * r;
}
vec3 uintToColorDebug(uint v) {
if (v == 0u) {
return vec3(0.0, 1.0, 0.0);
} else if (v == 1u) {
return vec3(0.0, 0.0, 1.0);
} else if (v == 2u) {
return vec3(1.0, 1.0, 0.0);
} else if (v == 3u) {
return vec3(1.0, 0.0, 0.0);
} else if (v == 4u) {
return vec3(1.0, 0.0, 1.0);
} else if (v == 5u) {
return vec3(0.0, 1.0, 1.0);
}
return vec3(0.0, 0.0, 0.0);
}
#if defined(TARGET_MOBILE)
#define MIN_PERCEPTUAL_ROUGHNESS 0.089
#define MIN_ROUGHNESS            0.007921
#else
#define MIN_PERCEPTUAL_ROUGHNESS 0.045
#define MIN_ROUGHNESS            0.002025
#endif
#define MIN_N_DOT_V 1e-4
float clampNoV(float NoV) {
return max(NoV, MIN_N_DOT_V);
}
vec3 computeDiffuseColor(const vec4 baseColor, float metallic) {
return baseColor.rgb * (1.0 - metallic);
}
vec3 computeF0(const vec4 baseColor, float metallic, float reflectance) {
return baseColor.rgb * metallic + (reflectance * (1.0 - metallic));
}
float computeDielectricF0(float reflectance) {
return 0.16 * reflectance * reflectance;
}
float computeMetallicFromSpecularColor(const vec3 specularColor) {
return max3(specularColor);
}
float computeRoughnessFromGlossiness(float glossiness) {
return 1.0 - glossiness;
}
float perceptualRoughnessToRoughness(float perceptualRoughness) {
return perceptualRoughness * perceptualRoughness;
}
float roughnessToPerceptualRoughness(float roughness) {
return sqrt(roughness);
}
float iorToF0(float transmittedIor, float incidentIor) {
return sq((transmittedIor - incidentIor) / (transmittedIor + incidentIor));
}
float f0ToIor(float f0) {
float r = sqrt(f0);
return (1.0 + r) / (1.0 - r);
}
vec3 f0ClearCoatToSurface(const vec3 f0) {
#if FILAMENT_QUALITY == FILAMENT_QUALITY_LOW
return saturate(f0 * (f0 * 0.526868 + 0.529324) - 0.0482256);
#else
return saturate(f0 * (f0 * (0.941892 - 0.263008 * f0) + 0.346479) - 0.0285998);
#endif
}

highp mat4 getViewFromWorldMatrix() {
return frameUniforms.viewFromWorldMatrix;
}
highp mat4 getWorldFromViewMatrix() {
return frameUniforms.worldFromViewMatrix;
}
highp mat4 getClipFromViewMatrix() {
return frameUniforms.clipFromViewMatrix;
}
highp mat4 getViewFromClipMatrix() {
return frameUniforms.viewFromClipMatrix;
}
highp mat4 getClipFromWorldMatrix() {
return frameUniforms.clipFromWorldMatrix;
}
highp mat4 getWorldFromClipMatrix() {
return frameUniforms.worldFromClipMatrix;
}
float getTime() {
return frameUniforms.time;
}
highp vec4 getUserTime() {
return frameUniforms.userTime;
}
highp float getUserTimeMod(float m) {
return mod(mod(frameUniforms.userTime.x, m) + mod(frameUniforms.userTime.y, m), m);
}
highp vec2 uvToRenderTargetUV(const highp vec2 uv) {
#if defined(TARGET_METAL_ENVIRONMENT) || defined(TARGET_VULKAN_ENVIRONMENT)
return vec2(uv.x, 1.0 - uv.y);
#else
return uv;
#endif
}
#define FILAMENT_OBJECT_SKINNING_ENABLED_BIT   0x100u
#define FILAMENT_OBJECT_MORPHING_ENABLED_BIT   0x200u
#define FILAMENT_OBJECT_CONTACT_SHADOWS_BIT    0x400u
highp vec4 getResolution() {
return frameUniforms.resolution;
}
highp vec3 getWorldCameraPosition() {
return frameUniforms.cameraPosition;
}
highp vec3 getWorldOffset() {
return frameUniforms.worldOffset;
}
float getExposure() {
return frameUniforms.exposure;
}
float getEV100() {
return frameUniforms.ev100;
}

float getObjectUserData() {
return object_uniforms.userData;
}
#if defined(HAS_ATTRIBUTE_COLOR)
vec4 getColor() {
return vertex_color;
}
#endif
#if defined(HAS_ATTRIBUTE_UV0)
highp vec2 getUV0() {
return vertex_uv01.xy;
}
#endif
#if defined(HAS_ATTRIBUTE_UV1)
highp vec2 getUV1() {
return vertex_uv01.zw;
}
#endif
#if defined(BLEND_MODE_MASKED)
float getMaskThreshold() {
return materialParams._maskThreshold;
}
#endif
highp mat3 getWorldTangentFrame() {
return shading_tangentToWorld;
}
highp vec3 getWorldPosition() {
return shading_position;
}
vec3 getWorldViewVector() {
return shading_view;
}
vec3 getWorldNormalVector() {
return shading_normal;
}
vec3 getWorldGeometricNormalVector() {
return shading_geometricNormal;
}
vec3 getWorldReflectedVector() {
return shading_reflected;
}
float getNdotV() {
return shading_NoV;
}
highp vec3 getNormalizedPhysicalViewportCoord() {
return vec3(shading_normalizedViewportCoord, gl_FragCoord.z);
}
highp vec3 getNormalizedViewportCoord() {
highp vec2 scale = frameUniforms.logicalViewportScale;
highp vec2 offset = frameUniforms.logicalViewportOffset;
highp vec2 logicalUv = shading_normalizedViewportCoord * scale + offset;
return vec3(logicalUv, gl_FragCoord.z);
}
#if defined(VARIANT_HAS_SHADOWING) && defined(VARIANT_HAS_DYNAMIC_LIGHTING)
highp vec4 getSpotLightSpacePosition(uint index, highp vec3 dir, highp float zLight) {
highp mat4 lightFromWorldMatrix = shadowUniforms.shadows[index].lightFromWorldMatrix;
float bias = shadowUniforms.shadows[index].normalBias * zLight;
return computeLightSpacePosition(getWorldPosition(), getWorldNormalVector(),
dir, bias, lightFromWorldMatrix);
}
#endif
#if defined(MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY)
bool isDoubleSided() {
return materialParams._doubleSided;
}
#endif
uint getShadowCascade() {
vec3 viewPos = mulMat4x4Float3(getViewFromWorldMatrix(), getWorldPosition()).xyz;
bvec4 greaterZ = greaterThan(frameUniforms.cascadeSplits, vec4(viewPos.z));
uint cascadeCount = frameUniforms.cascades & 0xFu;
return clamp(uint(dot(vec4(greaterZ), vec4(1.0))), 0u, cascadeCount - 1u);
}
#if defined(VARIANT_HAS_SHADOWING) && defined(VARIANT_HAS_DIRECTIONAL_LIGHTING)
highp vec4 getCascadeLightSpacePosition(uint cascade) {
if (cascade == 0u) {
return vertex_lightSpacePosition;
}
return computeLightSpacePosition(getWorldPosition(), getWorldNormalVector(),
frameUniforms.lightDirection,
shadowUniforms.shadows[cascade].normalBias,
shadowUniforms.shadows[cascade].lightFromWorldMatrix);
}
#endif

#if defined(SHADING_MODEL_CLOTH)
#if !defined(MATERIAL_HAS_SUBSURFACE_COLOR)
#define MATERIAL_CAN_SKIP_LIGHTING
#endif
#elif defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
#else
#define MATERIAL_CAN_SKIP_LIGHTING
#endif
struct MaterialInputs {
vec4  baseColor;
#if !defined(SHADING_MODEL_UNLIT)
#if !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
float roughness;
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
float metallic;
float reflectance;
#endif
float ambientOcclusion;
#endif
vec4  emissive;
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
vec3 sheenColor;
float sheenRoughness;
#endif
float clearCoat;
float clearCoatRoughness;
float anisotropy;
vec3  anisotropyDirection;
#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
float thickness;
#endif
#if defined(SHADING_MODEL_SUBSURFACE)
float subsurfacePower;
vec3  subsurfaceColor;
#endif
#if defined(SHADING_MODEL_CLOTH)
vec3  sheenColor;
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
vec3  subsurfaceColor;
#endif
#endif
#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
vec3  specularColor;
float glossiness;
#endif
#if defined(MATERIAL_HAS_NORMAL)
vec3  normal;
#endif
#if defined(MATERIAL_HAS_BENT_NORMAL)
vec3  bentNormal;
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
vec3  clearCoatNormal;
#endif
#if defined(MATERIAL_HAS_POST_LIGHTING_COLOR)
vec4  postLightingColor;
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
#if defined(MATERIAL_HAS_REFRACTION)
#if defined(MATERIAL_HAS_ABSORPTION)
vec3 absorption;
#endif
#if defined(MATERIAL_HAS_TRANSMISSION)
float transmission;
#endif
#if defined(MATERIAL_HAS_IOR)
float ior;
#endif
#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
float microThickness;
#endif
#elif !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
#if defined(MATERIAL_HAS_IOR)
float ior;
#endif
#endif
#endif
};
void initMaterial(out MaterialInputs material) {
material.baseColor = vec4(1.0);
#if !defined(SHADING_MODEL_UNLIT)
#if !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
material.roughness = 1.0;
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
material.metallic = 0.0;
material.reflectance = 0.5;
#endif
material.ambientOcclusion = 1.0;
#endif
material.emissive = vec4(vec3(0.0), 1.0);
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
#if defined(MATERIAL_HAS_SHEEN_COLOR)
material.sheenColor = vec3(0.0);
material.sheenRoughness = 0.0;
#endif
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT)
material.clearCoat = 1.0;
material.clearCoatRoughness = 0.0;
#endif
#if defined(MATERIAL_HAS_ANISOTROPY)
material.anisotropy = 0.0;
material.anisotropyDirection = vec3(1.0, 0.0, 0.0);
#endif
#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
material.thickness = 0.5;
#endif
#if defined(SHADING_MODEL_SUBSURFACE)
material.subsurfacePower = 12.234;
material.subsurfaceColor = vec3(1.0);
#endif
#if defined(SHADING_MODEL_CLOTH)
material.sheenColor = sqrt(material.baseColor.rgb);
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
material.subsurfaceColor = vec3(0.0);
#endif
#endif
#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
material.glossiness = 0.0;
material.specularColor = vec3(0.0);
#endif
#if defined(MATERIAL_HAS_NORMAL)
material.normal = vec3(0.0, 0.0, 1.0);
#endif
#if defined(MATERIAL_HAS_BENT_NORMAL)
material.bentNormal = vec3(0.0, 0.0, 1.0);
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
material.clearCoatNormal = vec3(0.0, 0.0, 1.0);
#endif
#if defined(MATERIAL_HAS_POST_LIGHTING_COLOR)
material.postLightingColor = vec4(0.0);
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
#if defined(MATERIAL_HAS_REFRACTION)
#if defined(MATERIAL_HAS_ABSORPTION)
material.absorption = vec3(0.0);
#endif
#if defined(MATERIAL_HAS_TRANSMISSION)
material.transmission = 1.0;
#endif
#if defined(MATERIAL_HAS_IOR)
material.ior = 1.5;
#endif
#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
material.microThickness = 0.0;
#endif
#elif !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
#if defined(MATERIAL_HAS_IOR)
material.ior = 1.5;
#endif
#endif
#endif
}
#if defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
struct LightData {
vec4  colorIntensity;
vec3  l;
float NdotL;
vec3  worldPosition;
float attenuation;
float visibility;
};
struct ShadingData {
vec3  diffuseColor;
float perceptualRoughness;
vec3  f0;
float roughness;
};
#endif

void computeShadingParams() {
#if defined(HAS_ATTRIBUTE_TANGENTS)
highp vec3 n = vertex_worldNormal;
#if defined(MATERIAL_NEEDS_TBN)
highp vec3 t = vertex_worldTangent.xyz;
highp vec3 b = cross(n, t) * sign(vertex_worldTangent.w);
#endif
#if defined(MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY)
if (isDoubleSided()) {
n = gl_FrontFacing ? n : -n;
#if defined(MATERIAL_NEEDS_TBN)
t = gl_FrontFacing ? t : -t;
b = gl_FrontFacing ? b : -b;
#endif
}
#endif
shading_geometricNormal = normalize(n);
#if defined(MATERIAL_NEEDS_TBN)
shading_tangentToWorld = mat3(t, b, n);
#endif
#endif
shading_position = vertex_worldPosition.xyz;
shading_view = normalize(frameUniforms.cameraPosition - shading_position);
shading_normalizedViewportCoord = vertex_position.xy * (0.5 / vertex_position.w) + 0.5;
}
void prepareMaterial(const MaterialInputs material) {
#if defined(HAS_ATTRIBUTE_TANGENTS)
#if defined(MATERIAL_HAS_NORMAL)
shading_normal = normalize(shading_tangentToWorld * material.normal);
#else
shading_normal = getWorldGeometricNormalVector();
#endif
shading_NoV = clampNoV(dot(shading_normal, shading_view));
shading_reflected = reflect(-shading_view, shading_normal);
#if defined(MATERIAL_HAS_BENT_NORMAL)
shading_bentNormal = normalize(shading_tangentToWorld * material.bentNormal);
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT)
#if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
shading_clearCoatNormal = normalize(shading_tangentToWorld * material.clearCoatNormal);
#else
shading_clearCoatNormal = getWorldGeometricNormalVector();
#endif
#endif
#endif
}

vec4 fog(vec4 color, vec3 view) {
if (frameUniforms.fogDensity > 0.0) {
float A = frameUniforms.fogDensity;
float B = frameUniforms.fogHeightFalloff;
float d = length(view);
float h = max(0.001, view.y);
float fogIntegralFunctionOfDistance = A * ((1.0 - exp(-B * h)) / h);
float fogIntegral = fogIntegralFunctionOfDistance * max(d - frameUniforms.fogStart, 0.0);
float fogOpacity = max(1.0 - exp2(-fogIntegral), 0.0);
fogOpacity = min(fogOpacity, frameUniforms.fogMaxOpacity);
vec3 fogColor = frameUniforms.fogColor;
if (frameUniforms.fogColorFromIbl > 0.0) {
float lod = frameUniforms.iblRoughnessOneLevel;
fogColor *= textureLod(light_iblSpecular, view, lod).rgb * frameUniforms.iblLuminance;
}
fogColor *= fogOpacity;
if (frameUniforms.fogInscatteringSize > 0.0) {
float inscatteringIntegral = fogIntegralFunctionOfDistance *
max(d - frameUniforms.fogInscatteringStart, 0.0);
float inscatteringOpacity = max(1.0 - exp2(-inscatteringIntegral), 0.0);
vec3 sunColor = frameUniforms.lightColorIntensity.rgb * frameUniforms.lightColorIntensity.w;
float sunAmount = max(dot(view, frameUniforms.lightDirection) / d, 0.0);
float sunInscattering = pow(sunAmount, frameUniforms.fogInscatteringSize);
fogColor += sunColor * (sunInscattering * inscatteringOpacity);
}
#if   defined(BLEND_MODE_OPAQUE)
#elif defined(BLEND_MODE_TRANSPARENT)
fogColor *= color.a;
#elif defined(BLEND_MODE_ADD)
fogColor = vec3(0.0);
#elif defined(BLEND_MODE_MASKED)
#elif defined(BLEND_MODE_MULTIPLY)
#elif defined(BLEND_MODE_SCREEN)
#endif
color.rgb = color.rgb * (1.0 - fogOpacity) + fogColor;
}
return color;
}
#line 31
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 30 "skybox.mat"
#endif

    void material(inout MaterialInputs material) {
        prepareMaterial(material);
        vec4 sky;
        if (materialParams.constantColor != 0) {
            sky = materialParams.color;
        } else {
            sky = vec4(textureLod(materialParams_skybox, variable_eyeDirection.xyz, 0.0).rgb, 1.0);
            sky.rgb *= frameUniforms.iblLuminance;
        }
        if (materialParams.showSun != 0 && frameUniforms.sun.w >= 0.0f) {
            vec3 direction = normalize(variable_eyeDirection.xyz);
            // Assume the sun is a sphere
            vec3 sun = frameUniforms.lightColorIntensity.rgb *
                    (frameUniforms.lightColorIntensity.a * (4.0 * PI));
            float cosAngle = dot(direction, frameUniforms.lightDirection);
            float x = (cosAngle - frameUniforms.sun.x) * frameUniforms.sun.z;
            float gradient = pow(1.0 - saturate(x), frameUniforms.sun.w);
            sky.rgb = sky.rgb + gradient * sun;
        }
        material.baseColor = sky;
    }
#line 885
void addEmissive(const MaterialInputs material, inout vec4 color) {
#if defined(MATERIAL_HAS_EMISSIVE)
highp vec4 emissive = material.emissive;
highp float attenuation = mix(1.0, getExposure(), emissive.w);
color.rgb += emissive.rgb * (attenuation * color.a);
#endif
}
#if defined(BLEND_MODE_MASKED)
float computeMaskedAlpha(float a) {
return (a - getMaskThreshold()) / max(fwidth(a), 1e-3) + 0.5;
}
#endif
vec4 evaluateMaterial(const MaterialInputs material) {
vec4 color = material.baseColor;
#if defined(BLEND_MODE_MASKED)
color.a = computeMaskedAlpha(color.a);
if (color.a <= 0.0) {
discard;
}
if (frameUniforms.needsAlphaChannel == 1.0) {
color.a = 1.0;
}
#endif
#if defined(VARIANT_HAS_DIRECTIONAL_LIGHTING)
#if defined(VARIANT_HAS_SHADOWING)
float visibility = 1.0;
uint cascade = getShadowCascade();
bool cascadeHasVisibleShadows = bool(frameUniforms.cascades & ((1u << cascade) << 8u));
bool hasDirectionalShadows = bool(frameUniforms.directionalShadows & 1u);
if (hasDirectionalShadows && cascadeHasVisibleShadows) {
highp vec4 shadowPosition = getShadowPosition(cascade);
visibility = shadow(true, light_shadowMap, cascade, shadowPosition, 0.0f);
}
if ((frameUniforms.directionalShadows & 0x2u) != 0u && visibility > 0.0) {
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_CONTACT_SHADOWS_BIT) != 0u) {
visibility *= (1.0 - screenSpaceContactShadow(frameUniforms.lightDirection));
}
}
color *= 1.0 - visibility;
#else
color = vec4(0.0);
#endif
#elif defined(MATERIAL_HAS_SHADOW_MULTIPLIER)
color = vec4(0.0);
#endif
addEmissive(material, color);
return color;
}
layout(location = 0) out vec4 fragColor;
#if defined(MATERIAL_HAS_POST_LIGHTING_COLOR)
void blendPostLightingColor(const MaterialInputs material, inout vec4 color) {
#if defined(POST_LIGHTING_BLEND_MODE_OPAQUE)
color = material.postLightingColor;
#elif defined(POST_LIGHTING_BLEND_MODE_TRANSPARENT)
color = material.postLightingColor + color * (1.0 - material.postLightingColor.a);
#elif defined(POST_LIGHTING_BLEND_MODE_ADD)
color += material.postLightingColor;
#elif defined(POST_LIGHTING_BLEND_MODE_MULTIPLY)
color *= material.postLightingColor;
#elif defined(POST_LIGHTING_BLEND_MODE_SCREEN)
color += material.postLightingColor * (1.0 - color);
#endif
}
#endif
void main() {
filament_lodBias = frameUniforms.lodBias;
initObjectUniforms(object_uniforms);
computeShadingParams();
MaterialInputs inputs;
initMaterial(inputs);
material(inputs);
fragColor = evaluateMaterial(inputs);
#if defined(VARIANT_HAS_DIRECTIONAL_LIGHTING) && defined(VARIANT_HAS_SHADOWING)
bool visualizeCascades = bool(frameUniforms.cascades & 0x10u);
if (visualizeCascades) {
fragColor.rgb *= uintToColorDebug(getShadowCascade());
}
#endif
#if defined(VARIANT_HAS_FOG)
vec3 view = getWorldPosition() - getWorldCameraPosition();
fragColor = fog(fragColor, view);
#endif
#if defined(MATERIAL_HAS_POST_LIGHTING_COLOR) && !defined(MATERIAL_HAS_REFLECTIONS)
blendPostLightingColor(inputs, fragColor);
#endif
}

