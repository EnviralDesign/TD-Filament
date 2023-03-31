#version 450 core

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
#define GEOMETRIC_SPECULAR_AA
#define CLEAR_COAT_IOR_CHANGE
#define SPECULAR_AMBIENT_OCCLUSION 1
#define MATERIAL_HAS_REFRACTION
#define REFRACTION_MODE_CUBEMAP 1
#define REFRACTION_MODE_SCREEN_SPACE 2
#define REFRACTION_MODE REFRACTION_MODE_CUBEMAP
#define REFRACTION_TYPE_SOLID 0
#define REFRACTION_TYPE_THIN 1
#define REFRACTION_TYPE REFRACTION_TYPE_SOLID
#define MULTI_BOUNCE_AMBIENT_OCCLUSION 1
#define VARIANT_HAS_DIRECTIONAL_LIGHTING
#define VARIANT_HAS_DYNAMIC_LIGHTING
#define VARIANT_HAS_SHADOWING
#define VARIANT_HAS_FOG
#define VARIANT_HAS_VSM
#define MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY
#define BLEND_MODE_OPAQUE
#define POST_LIGHTING_BLEND_MODE_TRANSPARENT
#define SHADING_MODEL_SUBSURFACE
#define SHADING_INTERPOLATION 
#define HAS_ATTRIBUTE_TANGENTS
#define HAS_ATTRIBUTE_UV0
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

layout(binding = 0, std140) uniform FrameUniforms {
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

layout(binding = 1, std140) uniform ObjectUniforms {
    PerRenderableData data[CONFIG_MAX_INSTANCES];
} objectUniforms;

layout(binding = 4, std140) uniform LightsUniforms {
    mat4 lights[256];
} lightsUniforms;

layout(binding = 5, std140) uniform ShadowUniforms {
    ShadowData shadows[64];
} shadowUniforms;

layout(binding = 6, std140) uniform FroxelRecordUniforms {
    uvec4 records[1024];
} froxelRecordUniforms;

layout(binding = 7, std140) uniform MaterialParams {
    vec3 baseColor;
    float metallic;
    float roughness;
    float reflectance;
    vec3 sheenColor;
    float sheenRoughness;
    float clearCoat;
    float clearCoatRoughness;
    float anisotropy;
    vec3 anisotropyDirection;
    float ambientOcclusion;
    vec3 normal;
    vec3 bentNormal;
    vec3 clearCoatNormal;
    vec4 emissive;
    vec4 postLightingColor;
    float ior;
    float transmission;
    vec3 absorption;
    float thickness;
    float microThickness;
    lowp float _specularAntiAliasingVariance;
    lowp float _specularAntiAliasingThreshold;
    bool _doubleSided;
} materialParams;

uniform highp sampler2DArray light_shadowMap;
uniform mediump usampler2D light_froxels;
uniform mediump sampler2D light_iblDFG;
uniform mediump samplerCube light_iblSpecular;
uniform mediump sampler2DArray light_ssao;
uniform mediump sampler2DArray light_ssr;
uniform highp sampler2D light_structure;

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
#line 149
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 148 "subsurface.mat"
#endif

    void material(inout MaterialInputs material) {

        // CUSTOM_FRAGMENT_CODE_STARTS_HERE
        // NOTE: This string comment above is used for parsing downstream, do not modifiy or delete.

        //material.normal = materialParams.normal;
        //material.bentNormal = materialParams.bentNormal;
        //material.clearCoatNormal = materialParams.clearCoatNormal;
        
        prepareMaterial(material);

        //material.baseColor.rgb = materialParams.baseColor;
        //material.metallic = materialParams.metallic;
        //material.roughness = materialParams.roughness;
        //material.reflectance = materialParams.reflectance;
        //material.sheenColor = materialParams.sheenColor;
        //material.sheenRoughness = materialParams.sheenRoughness;
        //material.clearCoat = materialParams.clearCoat;
        //material.clearCoatRoughness = materialParams.clearCoatRoughness;
        //material.anisotropy = materialParams.anisotropy;
        //material.anisotropyDirection = materialParams.anisotropyDirection;
        //material.ambientOcclusion = materialParams.ambientOcclusion;
        //material.emissive = materialParams.emissive;
        //material.postLightingColor = materialParams.postLightingColor;
        //material.ior = materialParams.ior;
        //material.transmission = materialParams.transmission;
        //material.absorption = materialParams.absorption;
        //material.thickness = materialParams.thickness;
        //material.microThickness = materialParams.microThickness;

        // NOTE: This string comment below is used for parsing downstream, do not modifiy or delete.
        // CUSTOM_FRAGMENT_CODE_ENDS_HERE

    }
#line 932
struct Light {
vec4 colorIntensity;
vec3 l;
float attenuation;
highp vec3 worldPosition;
float NoL;
highp vec3 direction;
float zLight;
bool castsShadows;
bool contactShadows;
uint type;
uint shadowIndex;
uint channels;
};
struct PixelParams {
vec3  diffuseColor;
float perceptualRoughness;
float perceptualRoughnessUnclamped;
vec3  f0;
float roughness;
vec3  dfg;
vec3  energyCompensation;
#if defined(MATERIAL_HAS_CLEAR_COAT)
float clearCoat;
float clearCoatPerceptualRoughness;
float clearCoatRoughness;
#endif
#if defined(MATERIAL_HAS_SHEEN_COLOR)
vec3  sheenColor;
#if !defined(SHADING_MODEL_CLOTH)
float sheenRoughness;
float sheenPerceptualRoughness;
float sheenScaling;
float sheenDFG;
#endif
#endif
#if defined(MATERIAL_HAS_ANISOTROPY)
vec3  anisotropicT;
vec3  anisotropicB;
float anisotropy;
#endif
#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
float thickness;
#endif
#if defined(SHADING_MODEL_SUBSURFACE)
vec3  subsurfaceColor;
float subsurfacePower;
#endif
#if defined(SHADING_MODEL_CLOTH) && defined(MATERIAL_HAS_SUBSURFACE_COLOR)
vec3  subsurfaceColor;
#endif
#if defined(MATERIAL_HAS_REFRACTION)
float etaRI;
float etaIR;
float transmission;
float uThickness;
vec3  absorption;
#endif
};
float computeMicroShadowing(float NoL, float visibility) {
float aperture = inversesqrt(1.0 - min(visibility, 0.9999));
float microShadow = saturate(NoL * aperture);
return microShadow * microShadow;
}
vec3 getReflectedVector(const PixelParams pixel, const vec3 v, const vec3 n) {
#if defined(MATERIAL_HAS_ANISOTROPY)
vec3  anisotropyDirection = pixel.anisotropy >= 0.0 ? pixel.anisotropicB : pixel.anisotropicT;
vec3  anisotropicTangent  = cross(anisotropyDirection, v);
vec3  anisotropicNormal   = cross(anisotropicTangent, anisotropyDirection);
float bendFactor          = abs(pixel.anisotropy) * saturate(5.0 * pixel.perceptualRoughness);
vec3  bentNormal          = normalize(mix(n, anisotropicNormal, bendFactor));
vec3 r = reflect(-v, bentNormal);
#else
vec3 r = reflect(-v, n);
#endif
return r;
}
void getAnisotropyPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_ANISOTROPY)
vec3 direction = material.anisotropyDirection;
pixel.anisotropy = material.anisotropy;
pixel.anisotropicT = normalize(shading_tangentToWorld * direction);
pixel.anisotropicB = normalize(cross(getWorldGeometricNormalVector(), pixel.anisotropicT));
#endif
}

#define SHADOW_SAMPLING_RUNTIME_PCF     0u
#define SHADOW_SAMPLING_RUNTIME_EVSM    1u
#define SHADOW_SAMPLING_RUNTIME_DPCF    2u
#define SHADOW_SAMPLING_RUNTIME_PCSS    3u
#define SHADOW_SAMPLING_PCF_HARD        0
#define SHADOW_SAMPLING_PCF_LOW         1
#define SHADOW_SAMPLING_METHOD          SHADOW_SAMPLING_PCF_LOW
float sampleDepth(const mediump sampler2DArrayShadow map,
const highp vec4 scissorNormalized,
const uint layer,  highp vec2 uv, float depth) {
uv = clamp(uv, scissorNormalized.xy, scissorNormalized.zw);
return texture(map, vec4(uv, layer, saturate(depth)));
}
#if SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_HARD
float ShadowSample_PCF_Hard(const mediump sampler2DArrayShadow map,
const highp vec4 scissorNormalized,
const uint layer, const highp vec4 shadowPosition) {
highp vec3 position = shadowPosition.xyz * (1.0 / shadowPosition.w);
return sampleDepth(map, scissorNormalized, layer, position.xy, position.z);
}
#endif
#if SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_LOW
float ShadowSample_PCF_Low(const mediump sampler2DArrayShadow map,
const highp vec4 scissorNormalized,
const uint layer, const highp vec4 shadowPosition) {
highp vec3 position = shadowPosition.xyz * (1.0 / shadowPosition.w);
highp vec2 size = vec2(textureSize(map, 0));
highp vec2 texelSize = vec2(1.0) / size;
float depth = position.z;
position.xy = clamp(position.xy, vec2(-1.0), vec2(2.0));
vec2 offset = vec2(0.5);
highp vec2 uv = (position.xy * size) + offset;
highp vec2 base = (floor(uv) - offset) * texelSize;
highp vec2 st = fract(uv);
vec2 uw = vec2(3.0 - 2.0 * st.x, 1.0 + 2.0 * st.x);
vec2 vw = vec2(3.0 - 2.0 * st.y, 1.0 + 2.0 * st.y);
highp vec2 u = vec2((2.0 - st.x) / uw.x - 1.0, st.x / uw.y + 1.0);
highp vec2 v = vec2((2.0 - st.y) / vw.x - 1.0, st.y / vw.y + 1.0);
u *= texelSize.x;
v *= texelSize.y;
float sum = 0.0;
sum += uw.x * vw.x * sampleDepth(map, scissorNormalized, layer, base + vec2(u.x, v.x), depth);
sum += uw.y * vw.x * sampleDepth(map, scissorNormalized, layer, base + vec2(u.y, v.x), depth);
sum += uw.x * vw.y * sampleDepth(map, scissorNormalized, layer, base + vec2(u.x, v.y), depth);
sum += uw.y * vw.y * sampleDepth(map, scissorNormalized, layer, base + vec2(u.y, v.y), depth);
return sum * (1.0 / 16.0);
}
#endif
float ShadowSample_PCF(const mediump sampler2DArray map,
const highp vec4 scissorNormalized,
const uint layer, const highp vec4 shadowPosition) {
highp vec3 position = shadowPosition.xyz * (1.0 / shadowPosition.w);
highp vec2 size = vec2(textureSize(map, 0));
highp vec2 tc = clamp(position.xy, scissorNormalized.xy, scissorNormalized.zw);
highp vec2 st = tc.xy * size - 0.5;
vec4 d;
#if defined(FILAMENT_HAS_FEATURE_TEXTURE_GATHER)
d = textureGather(map, vec3(tc, layer), 0);
#else
d[0] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(0, 1)).r;
d[1] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(1, 1)).r;
d[2] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(1, 0)).r;
d[3] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(0, 0)).r;
#endif
vec4 pcf = step(0.0, position.zzzz - d);
highp vec2 grad = fract(st);
return mix(mix(pcf.w, pcf.z, grad.x), mix(pcf.x, pcf.y, grad.x), grad.y);
}
mediump vec2 poissonDisk[64] = vec2[](
vec2(0.511749, 0.547686), vec2(0.58929, 0.257224), vec2(0.165018, 0.57663), vec2(0.407692, 0.742285),
vec2(0.707012, 0.646523), vec2(0.31463, 0.466825), vec2(0.801257, 0.485186), vec2(0.418136, 0.146517),
vec2(0.579889, 0.0368284), vec2(0.79801, 0.140114), vec2(-0.0413185, 0.371455), vec2(-0.0529108, 0.627352),
vec2(0.0821375, 0.882071), vec2(0.17308, 0.301207), vec2(-0.120452, 0.867216), vec2(0.371096, 0.916454),
vec2(-0.178381, 0.146101), vec2(-0.276489, 0.550525), vec2(0.12542, 0.126643), vec2(-0.296654, 0.286879),
vec2(0.261744, -0.00604975), vec2(-0.213417, 0.715776), vec2(0.425684, -0.153211), vec2(-0.480054, 0.321357),
vec2(-0.0717878, -0.0250567), vec2(-0.328775, -0.169666), vec2(-0.394923, 0.130802), vec2(-0.553681, -0.176777),
vec2(-0.722615, 0.120616), vec2(-0.693065, 0.309017), vec2(0.603193, 0.791471), vec2(-0.0754941, -0.297988),
vec2(0.109303, -0.156472), vec2(0.260605, -0.280111), vec2(0.129731, -0.487954), vec2(-0.537315, 0.520494),
vec2(-0.42758, 0.800607), vec2(0.77309, -0.0728102), vec2(0.908777, 0.328356), vec2(0.985341, 0.0759158),
vec2(0.947536, -0.11837), vec2(-0.103315, -0.610747), vec2(0.337171, -0.584), vec2(0.210919, -0.720055),
vec2(0.41894, -0.36769), vec2(-0.254228, -0.49368), vec2(-0.428562, -0.404037), vec2(-0.831732, -0.189615),
vec2(-0.922642, 0.0888026), vec2(-0.865914, 0.427795), vec2(0.706117, -0.311662), vec2(0.545465, -0.520942),
vec2(-0.695738, 0.664492), vec2(0.389421, -0.899007), vec2(0.48842, -0.708054), vec2(0.760298, -0.62735),
vec2(-0.390788, -0.707388), vec2(-0.591046, -0.686721), vec2(-0.769903, -0.413775), vec2(-0.604457, -0.502571),
vec2(-0.557234, 0.00451362), vec2(0.147572, -0.924353), vec2(-0.0662488, -0.892081), vec2(0.863832, -0.407206)
);
const uint DPCF_SHADOW_TAP_COUNT                = 12u;
const uint PCSS_SHADOW_BLOCKER_SEARCH_TAP_COUNT = 16u;
const uint PCSS_SHADOW_FILTER_TAP_COUNT         = 16u;
float hardenedKernel(float x) {
x = 2.0 * x - 1.0;
float s = sign(x);
x = 1.0 - s * x;
x = x * x * x;
x = s - x * s;
return 0.5 * x + 0.5;
}
highp vec2 computeReceiverPlaneDepthBias(const highp vec3 position) {
highp vec3 duvz_dx = dFdx(position);
highp vec3 duvz_dy = dFdy(position);
highp vec2 dz_duv = inverse(transpose(mat2(duvz_dx.xy, duvz_dy.xy))) * vec2(duvz_dx.z, duvz_dy.z);
return dz_duv;
}
mat2 getRandomRotationMatrix(highp vec2 fragCoord) {
fragCoord += vec2(frameUniforms.temporalNoise);
float randomAngle = interleavedGradientNoise(fragCoord) * (2.0 * PI);
vec2 randomBase = vec2(cos(randomAngle), sin(randomAngle));
mat2 R = mat2(randomBase.x, randomBase.y, -randomBase.y, randomBase.x);
return R;
}
float getPenumbraLs(const bool DIRECTIONAL, const uint index, const highp float zLight) {
float penumbra;
if (DIRECTIONAL) {
penumbra = shadowUniforms.shadows[index].bulbRadiusLs;
} else {
penumbra = shadowUniforms.shadows[index].bulbRadiusLs / zLight;
}
return penumbra;
}
float getPenumbraRatio(const bool DIRECTIONAL, const uint index,
float z_receiver, float z_blocker) {
float penumbraRatio;
if (DIRECTIONAL) {
penumbraRatio = (z_blocker - z_receiver) / (1.0 - z_blocker);
} else {
float nearOverFarMinusNear = shadowUniforms.shadows[index].nearOverFarMinusNear;
penumbraRatio = (nearOverFarMinusNear + z_blocker) / (nearOverFarMinusNear + z_receiver) - 1.0;
}
return penumbraRatio * frameUniforms.shadowPenumbraRatioScale;
}
void blockerSearchAndFilter(out float occludedCount, out float z_occSum,
const mediump sampler2DArray map, const highp vec4 scissorNormalized, const highp vec2 uv,
const float z_rec, const uint layer,
const highp vec2 filterRadii, const mat2 R, const highp vec2 dz_duv,
const uint tapCount) {
occludedCount = 0.0;
z_occSum = 0.0;
for (uint i = 0u; i < tapCount; i++) {
highp vec2 duv = R * (poissonDisk[i] * filterRadii);
highp vec2 tc = clamp(uv + duv, scissorNormalized.xy, scissorNormalized.zw);
float z_occ = textureLod(map, vec3(tc, layer), 0.0).r;
float z_bias = dot(dz_duv, duv);
float dz = z_occ - z_rec;
float occluded = step(z_bias, dz);
occludedCount += occluded;
z_occSum += z_occ * occluded;
}
}
float filterPCSS(const mediump sampler2DArray map,
const highp vec4 scissorNormalized,
const highp vec2 size,
const highp vec2 uv, const float z_rec, const uint layer,
const highp vec2 filterRadii, const mat2 R, const highp vec2 dz_duv,
const uint tapCount) {
float occludedCount = 0.0;
for (uint i = 0u; i < tapCount; i++) {
highp vec2 duv = R * (poissonDisk[i] * filterRadii);
vec4 d;
highp vec2 tc = clamp(uv + duv, scissorNormalized.xy, scissorNormalized.zw);
highp vec2 st = tc.xy * size - 0.5;
highp vec2 grad = fract(st);
#if defined(FILAMENT_HAS_FEATURE_TEXTURE_GATHER)
d = textureGather(map, vec3(tc, layer), 0);
#else
d[0] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(0, 1)).r;
d[1] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(1, 1)).r;
d[2] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(1, 0)).r;
d[3] = texelFetchOffset(map, ivec3(st, layer), 0, ivec2(0, 0)).r;
#endif
float z_bias = dot(dz_duv, duv);
vec4 dz = d - vec4(z_rec);
vec4 pcf = step(z_bias, dz);
occludedCount += mix(mix(pcf.w, pcf.z, grad.x), mix(pcf.x, pcf.y, grad.x), grad.y);
}
return occludedCount * (1.0 / float(tapCount));
}
float ShadowSample_DPCF(const bool DIRECTIONAL,
const mediump sampler2DArray map,
const highp vec4 scissorNormalized,
const uint layer, const uint index,
const highp vec4 shadowPosition, const highp float zLight) {
highp vec3 position = shadowPosition.xyz * (1.0 / shadowPosition.w);
highp vec2 texelSize = vec2(1.0) / vec2(textureSize(map, 0));
highp vec2 dz_duv = computeReceiverPlaneDepthBias(position);
float penumbra = getPenumbraLs(DIRECTIONAL, index, zLight);
mat2 R = getRandomRotationMatrix(gl_FragCoord.xy);
float occludedCount = 0.0;
float z_occSum = 0.0;
blockerSearchAndFilter(occludedCount, z_occSum,
map, scissorNormalized, position.xy, position.z, layer, texelSize * penumbra, R, dz_duv,
DPCF_SHADOW_TAP_COUNT);
if (z_occSum == 0.0) {
return 1.0;
}
float penumbraRatio = getPenumbraRatio(DIRECTIONAL, index, position.z, z_occSum / occludedCount);
penumbraRatio = saturate(penumbraRatio);
float percentageOccluded = occludedCount * (1.0 / float(DPCF_SHADOW_TAP_COUNT));
percentageOccluded = mix(hardenedKernel(percentageOccluded), percentageOccluded, penumbraRatio);
return 1.0 - percentageOccluded;
}
float ShadowSample_PCSS(const bool DIRECTIONAL,
const mediump sampler2DArray map,
const highp vec4 scissorNormalized,
const uint layer, const uint index,
const highp vec4 shadowPosition, const highp float zLight) {
highp vec2 size = vec2(textureSize(map, 0));
highp vec2 texelSize = vec2(1.0) / size;
highp vec3 position = shadowPosition.xyz * (1.0 / shadowPosition.w);
highp vec2 dz_duv = computeReceiverPlaneDepthBias(position);
float penumbra = getPenumbraLs(DIRECTIONAL, index, zLight);
mat2 R = getRandomRotationMatrix(gl_FragCoord.xy);
float occludedCount = 0.0;
float z_occSum = 0.0;
blockerSearchAndFilter(occludedCount, z_occSum,
map, scissorNormalized, position.xy, position.z, layer, texelSize * penumbra, R, dz_duv,
PCSS_SHADOW_BLOCKER_SEARCH_TAP_COUNT);
if (z_occSum == 0.0) {
return 1.0;
}
float penumbraRatio = getPenumbraRatio(DIRECTIONAL, index, position.z, z_occSum / occludedCount);
float percentageOccluded = filterPCSS(map, scissorNormalized, size,
position.xy, position.z, layer,
texelSize * (penumbra * penumbraRatio),
R, dz_duv, PCSS_SHADOW_FILTER_TAP_COUNT);
return 1.0 - percentageOccluded;
}
struct ScreenSpaceRay {
highp vec3 ssRayStart;
highp vec3 ssRayEnd;
highp vec3 ssViewRayEnd;
highp vec3 uvRayStart;
highp vec3 uvRay;
};
void initScreenSpaceRay(out ScreenSpaceRay ray, highp vec3 wsRayStart, vec3 wsRayDirection, float wsRayLength) {
highp mat4 worldToClip = getClipFromWorldMatrix();
highp mat4 viewToClip = getClipFromViewMatrix();
highp vec3 wsRayEnd = wsRayStart + wsRayDirection * wsRayLength;
highp vec4 csRayStart = worldToClip * vec4(wsRayStart, 1.0);
highp vec4 csRayEnd = worldToClip * vec4(wsRayEnd, 1.0);
highp vec4 csViewRayEnd = csRayStart + viewToClip * vec4(0.0, 0.0, wsRayLength, 0.0);
ray.ssRayStart = csRayStart.xyz * (1.0 / csRayStart.w);
ray.ssRayEnd = csRayEnd.xyz * (1.0 / csRayEnd.w);
ray.ssViewRayEnd = csViewRayEnd.xyz * (1.0 / csViewRayEnd.w);
highp vec3 uvRayEnd = vec3(ray.ssRayEnd.xy * 0.5 + 0.5, ray.ssRayEnd.z);
ray.uvRayStart = vec3(ray.ssRayStart.xy * 0.5 + 0.5, ray.ssRayStart.z);
ray.uvRay = uvRayEnd - ray.uvRayStart;
}
float screenSpaceContactShadow(vec3 lightDirection) {
float occlusion = 0.0;
uint kStepCount = (frameUniforms.directionalShadows >> 8u) & 0xFFu;
float kDistanceMax = frameUniforms.ssContactShadowDistance;
ScreenSpaceRay rayData;
initScreenSpaceRay(rayData, shading_position, lightDirection, kDistanceMax);
highp float dt = 1.0 / float(kStepCount);
highp float tolerance = abs(rayData.ssViewRayEnd.z - rayData.ssRayStart.z) * dt;
float dither = interleavedGradientNoise(gl_FragCoord.xy) - 0.5;
highp float t = dt * dither + dt;
highp vec3 ray;
for (uint i = 0u ; i < kStepCount ; i++, t += dt) {
ray = rayData.uvRayStart + rayData.uvRay * t;
highp float z = textureLod(light_structure, uvToRenderTargetUV(ray.xy), 0.0).r;
highp float dz = z - ray.z;
if (abs(tolerance - dz) < tolerance) {
occlusion = 1.0;
break;
}
}
vec2 fade = max(12.0 * abs(ray.xy - 0.5) - 5.0, 0.0);
occlusion *= saturate(1.0 - dot(fade, fade));
return occlusion;
}
float linstep(const float min, const float max, const float v) {
return clamp((v - min) / (max - min), 0.0, 1.0);
}
float reduceLightBleed(const float pMax, const float amount) {
return linstep(amount, 1.0, pMax);
}
float chebyshevUpperBound(const highp vec2 moments, const highp float mean,
const highp float minVariance, const float lightBleedReduction) {
highp float variance = moments.y - (moments.x * moments.x);
variance = max(variance, minVariance);
highp float d = mean - moments.x;
float pMax = variance / (variance + d * d);
pMax = reduceLightBleed(pMax, lightBleedReduction);
return mean <= moments.x ? 1.0 : pMax;
}
float evaluateShadowVSM(const highp vec2 moments, const highp float depth) {
highp float depthScale = frameUniforms.vsmDepthScale * depth;
highp float minVariance = depthScale * depthScale;
return chebyshevUpperBound(moments, depth, minVariance, frameUniforms.vsmLightBleedReduction);
}
float ShadowSample_VSM(const bool ELVSM, const mediump sampler2DArray shadowMap,
const highp vec4 scissorNormalized,
const uint layer, const highp vec4 shadowPosition) {
highp vec3 position = vec3(shadowPosition.xy * (1.0 / shadowPosition.w), shadowPosition.z);
highp vec4 moments = texture(shadowMap, vec3(position.xy, layer));
highp float depth = position.z;
depth = depth * 2.0 - 1.0;
depth = frameUniforms.vsmExponent * depth;
depth = exp(depth);
float p = evaluateShadowVSM(moments.xy, depth);
if (ELVSM) {
p = min(p, evaluateShadowVSM(moments.zw, -1.0 / depth));
}
return p;
}
#if defined(VARIANT_HAS_DIRECTIONAL_LIGHTING)
highp vec4 getShadowPosition(const uint cascade) {
return getCascadeLightSpacePosition(cascade);
}
#endif
#if defined(VARIANT_HAS_DYNAMIC_LIGHTING)
highp vec4 getShadowPosition(const uint index,  const highp vec3 dir, const highp float zLight) {
return getSpotLightSpacePosition(index, dir, zLight);
}
#endif
uint getPointLightFace(const highp vec3 r) {
highp vec4 tc;
highp float rx = abs(r.x);
highp float ry = abs(r.y);
highp float rz = abs(r.z);
highp float d = max(rx, max(ry, rz));
if (d == rx) {
return (r.x >= 0.0 ? 0u : 1u);
} else if (d == ry) {
return (r.y >= 0.0 ? 2u : 3u);
} else {
return (r.z >= 0.0 ? 4u : 5u);
}
}
float shadow(const bool DIRECTIONAL,
const mediump sampler2DArrayShadow shadowMap,
const uint index, highp vec4 shadowPosition, highp float zLight) {
highp vec4 scissorNormalized = shadowUniforms.shadows[index].scissorNormalized;
uint layer = shadowUniforms.shadows[index].layer;
#if SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_HARD
return ShadowSample_PCF_Hard(shadowMap, scissorNormalized, layer, shadowPosition);
#elif SHADOW_SAMPLING_METHOD == SHADOW_SAMPLING_PCF_LOW
return ShadowSample_PCF_Low(shadowMap, scissorNormalized, layer, shadowPosition);
#endif
}
float shadow(const bool DIRECTIONAL,
const mediump sampler2DArray shadowMap,
const uint index, highp vec4 shadowPosition, highp float zLight) {
highp vec4 scissorNormalized = shadowUniforms.shadows[index].scissorNormalized;
uint layer = shadowUniforms.shadows[index].layer;
if (frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_EVSM) {
bool elvsm = shadowUniforms.shadows[index].elvsm;
return ShadowSample_VSM(elvsm, shadowMap, scissorNormalized, layer,
shadowPosition);
}
if (frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_DPCF) {
return ShadowSample_DPCF(DIRECTIONAL, shadowMap, scissorNormalized, layer, index,
shadowPosition, zLight);
}
if (frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_PCSS) {
return ShadowSample_PCSS(DIRECTIONAL, shadowMap, scissorNormalized, layer, index,
shadowPosition, zLight);
}
if (frameUniforms.shadowSamplingType == SHADOW_SAMPLING_RUNTIME_PCF) {
return ShadowSample_PCF(shadowMap, scissorNormalized,
layer, shadowPosition);
}
return 0.0;
}

#define DIFFUSE_LAMBERT             0
#define DIFFUSE_BURLEY              1
#define SPECULAR_D_GGX              0
#define SPECULAR_D_GGX_ANISOTROPIC  0
#define SPECULAR_D_CHARLIE          0
#define SPECULAR_V_SMITH_GGX        0
#define SPECULAR_V_SMITH_GGX_FAST   1
#define SPECULAR_V_GGX_ANISOTROPIC  2
#define SPECULAR_V_KELEMEN          3
#define SPECULAR_V_NEUBELT          4
#define SPECULAR_F_SCHLICK          0
#define BRDF_DIFFUSE                DIFFUSE_LAMBERT
#if FILAMENT_QUALITY < FILAMENT_QUALITY_HIGH
#define BRDF_SPECULAR_D             SPECULAR_D_GGX
#define BRDF_SPECULAR_V             SPECULAR_V_SMITH_GGX_FAST
#define BRDF_SPECULAR_F             SPECULAR_F_SCHLICK
#else
#define BRDF_SPECULAR_D             SPECULAR_D_GGX
#define BRDF_SPECULAR_V             SPECULAR_V_SMITH_GGX
#define BRDF_SPECULAR_F             SPECULAR_F_SCHLICK
#endif
#define BRDF_CLEAR_COAT_D           SPECULAR_D_GGX
#define BRDF_CLEAR_COAT_V           SPECULAR_V_KELEMEN
#define BRDF_ANISOTROPIC_D          SPECULAR_D_GGX_ANISOTROPIC
#define BRDF_ANISOTROPIC_V          SPECULAR_V_GGX_ANISOTROPIC
#define BRDF_CLOTH_D                SPECULAR_D_CHARLIE
#define BRDF_CLOTH_V                SPECULAR_V_NEUBELT
float D_GGX(float roughness, float NoH, const vec3 h) {
#if defined(TARGET_MOBILE)
vec3 NxH = cross(shading_normal, h);
float oneMinusNoHSquared = dot(NxH, NxH);
#else
float oneMinusNoHSquared = 1.0 - NoH * NoH;
#endif
float a = NoH * roughness;
float k = roughness / (oneMinusNoHSquared + a * a);
float d = k * k * (1.0 / PI);
return saturateMediump(d);
}
float D_GGX_Anisotropic(float at, float ab, float ToH, float BoH, float NoH) {
float a2 = at * ab;
highp vec3 d = vec3(ab * ToH, at * BoH, a2 * NoH);
highp float d2 = dot(d, d);
float b2 = a2 / d2;
return a2 * b2 * b2 * (1.0 / PI);
}
float D_Charlie(float roughness, float NoH) {
float invAlpha  = 1.0 / roughness;
float cos2h = NoH * NoH;
float sin2h = max(1.0 - cos2h, 0.0078125);
return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);
}
float V_SmithGGXCorrelated(float roughness, float NoV, float NoL) {
float a2 = roughness * roughness;
float lambdaV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
float lambdaL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
float v = 0.5 / (lambdaV + lambdaL);
return saturateMediump(v);
}
float V_SmithGGXCorrelated_Fast(float roughness, float NoV, float NoL) {
float v = 0.5 / mix(2.0 * NoL * NoV, NoL + NoV, roughness);
return saturateMediump(v);
}
float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV,
float ToL, float BoL, float NoV, float NoL) {
float lambdaV = NoL * length(vec3(at * ToV, ab * BoV, NoV));
float lambdaL = NoV * length(vec3(at * ToL, ab * BoL, NoL));
float v = 0.5 / (lambdaV + lambdaL);
return saturateMediump(v);
}
float V_Kelemen(float LoH) {
return saturateMediump(0.25 / (LoH * LoH));
}
float V_Neubelt(float NoV, float NoL) {
return saturateMediump(1.0 / (4.0 * (NoL + NoV - NoL * NoV)));
}
vec3 F_Schlick(const vec3 f0, float f90, float VoH) {
return f0 + (f90 - f0) * pow5(1.0 - VoH);
}
vec3 F_Schlick(const vec3 f0, float VoH) {
float f = pow(1.0 - VoH, 5.0);
return f + f0 * (1.0 - f);
}
float F_Schlick(float f0, float f90, float VoH) {
return f0 + (f90 - f0) * pow5(1.0 - VoH);
}
float distribution(float roughness, float NoH, const vec3 h) {
#if BRDF_SPECULAR_D == SPECULAR_D_GGX
return D_GGX(roughness, NoH, h);
#endif
}
float visibility(float roughness, float NoV, float NoL) {
#if BRDF_SPECULAR_V == SPECULAR_V_SMITH_GGX
return V_SmithGGXCorrelated(roughness, NoV, NoL);
#elif BRDF_SPECULAR_V == SPECULAR_V_SMITH_GGX_FAST
return V_SmithGGXCorrelated_Fast(roughness, NoV, NoL);
#endif
}
vec3 fresnel(const vec3 f0, float LoH) {
#if BRDF_SPECULAR_F == SPECULAR_F_SCHLICK
#if FILAMENT_QUALITY == FILAMENT_QUALITY_LOW
return F_Schlick(f0, LoH);
#else
float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
return F_Schlick(f0, f90, LoH);
#endif
#endif
}
float distributionAnisotropic(float at, float ab, float ToH, float BoH, float NoH) {
#if BRDF_ANISOTROPIC_D == SPECULAR_D_GGX_ANISOTROPIC
return D_GGX_Anisotropic(at, ab, ToH, BoH, NoH);
#endif
}
float visibilityAnisotropic(float roughness, float at, float ab,
float ToV, float BoV, float ToL, float BoL, float NoV, float NoL) {
#if BRDF_ANISOTROPIC_V == SPECULAR_V_SMITH_GGX
return V_SmithGGXCorrelated(roughness, NoV, NoL);
#elif BRDF_ANISOTROPIC_V == SPECULAR_V_GGX_ANISOTROPIC
return V_SmithGGXCorrelated_Anisotropic(at, ab, ToV, BoV, ToL, BoL, NoV, NoL);
#endif
}
float distributionClearCoat(float roughness, float NoH, const vec3 h) {
#if BRDF_CLEAR_COAT_D == SPECULAR_D_GGX
return D_GGX(roughness, NoH, h);
#endif
}
float visibilityClearCoat(float LoH) {
#if BRDF_CLEAR_COAT_V == SPECULAR_V_KELEMEN
return V_Kelemen(LoH);
#endif
}
float distributionCloth(float roughness, float NoH) {
#if BRDF_CLOTH_D == SPECULAR_D_CHARLIE
return D_Charlie(roughness, NoH);
#endif
}
float visibilityCloth(float NoV, float NoL) {
#if BRDF_CLOTH_V == SPECULAR_V_NEUBELT
return V_Neubelt(NoV, NoL);
#endif
}
float Fd_Lambert() {
return 1.0 / PI;
}
float Fd_Burley(float roughness, float NoV, float NoL, float LoH) {
float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
float lightScatter = F_Schlick(1.0, f90, NoL);
float viewScatter  = F_Schlick(1.0, f90, NoV);
return lightScatter * viewScatter * (1.0 / PI);
}
float Fd_Wrap(float NoL, float w) {
return saturate((NoL + w) / sq(1.0 + w));
}
float diffuse(float roughness, float NoV, float NoL, float LoH) {
#if BRDF_DIFFUSE == DIFFUSE_LAMBERT
return Fd_Lambert();
#elif BRDF_DIFFUSE == DIFFUSE_BURLEY
return Fd_Burley(roughness, NoV, NoL, LoH);
#endif
}

vec3 surfaceShading(const PixelParams pixel, const Light light, float occlusion) {
vec3 h = normalize(shading_view + light.l);
float NoL = light.NoL;
float NoH = saturate(dot(shading_normal, h));
float LoH = saturate(dot(light.l, h));
vec3 Fr = vec3(0.0);
if (NoL > 0.0) {
float D = distribution(pixel.roughness, NoH, h);
float V = visibility(pixel.roughness, shading_NoV, NoL);
vec3  F = fresnel(pixel.f0, LoH);
Fr = (D * V) * F * pixel.energyCompensation;
}
vec3 Fd = pixel.diffuseColor * diffuse(pixel.roughness, shading_NoV, NoL, LoH);
vec3 color = (Fd + Fr) * (NoL * occlusion);
float scatterVoH = saturate(dot(shading_view, -light.l));
float forwardScatter = exp2(scatterVoH * pixel.subsurfacePower - pixel.subsurfacePower);
float backScatter = saturate(NoL * pixel.thickness + (1.0 - pixel.thickness)) * 0.5;
float subsurface = mix(backScatter, 1.0, forwardScatter) * (1.0 - pixel.thickness);
color += pixel.subsurfaceColor * (subsurface * Fd_Lambert());
return (color * light.colorIntensity.rgb) * (light.colorIntensity.w * light.attenuation);
}

#define SPECULAR_AO_OFF             0
#define SPECULAR_AO_SIMPLE          1
#define SPECULAR_AO_BENT_NORMALS    2
float unpack(vec2 depth) {
return (depth.x * (256.0 / 257.0) + depth.y * (1.0 / 257.0));
}
struct SSAOInterpolationCache {
highp vec4 weights;
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED) || defined(MATERIAL_HAS_REFLECTIONS)
highp vec2 uv;
#endif
};
float evaluateSSAO(inout SSAOInterpolationCache cache) {
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED)
if (frameUniforms.aoSamplingQualityAndEdgeDistance < 0.0) {
return 1.0;
}
if (frameUniforms.aoSamplingQualityAndEdgeDistance > 0.0) {
highp vec2 size = vec2(textureSize(light_ssao, 0));
#if defined(FILAMENT_HAS_FEATURE_TEXTURE_GATHER)
vec4 ao = textureGather(light_ssao, vec3(cache.uv, 0.0), 0);
vec4 dg = textureGather(light_ssao, vec3(cache.uv, 0.0), 1);
vec4 db = textureGather(light_ssao, vec3(cache.uv, 0.0), 2);
#else
vec3 s01 = textureLodOffset(light_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(0, 1)).rgb;
vec3 s11 = textureLodOffset(light_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(1, 1)).rgb;
vec3 s10 = textureLodOffset(light_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(1, 0)).rgb;
vec3 s00 = textureLodOffset(light_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(0, 0)).rgb;
vec4 ao = vec4(s01.r, s11.r, s10.r, s00.r);
vec4 dg = vec4(s01.g, s11.g, s10.g, s00.g);
vec4 db = vec4(s01.b, s11.b, s10.b, s00.b);
#endif
vec4 depths;
depths.x = unpack(vec2(dg.x, db.x));
depths.y = unpack(vec2(dg.y, db.y));
depths.z = unpack(vec2(dg.z, db.z));
depths.w = unpack(vec2(dg.w, db.w));
depths *= -frameUniforms.cameraFar;
vec2 f = fract(cache.uv * size - 0.5);
vec4 b;
b.x = (1.0 - f.x) * f.y;
b.y = f.x * f.y;
b.z = f.x * (1.0 - f.y);
b.w = (1.0 - f.x) * (1.0 - f.y);
highp mat4 m = getViewFromWorldMatrix();
highp float d = dot(vec3(m[0].z, m[1].z, m[2].z), shading_position) + m[3].z;
highp vec4 w = (vec4(d) - depths) * frameUniforms.aoSamplingQualityAndEdgeDistance;
w = max(vec4(MEDIUMP_FLT_MIN), 1.0 - w * w) * b;
cache.weights = w / (w.x + w.y + w.z + w.w);
return dot(ao, cache.weights);
} else {
return textureLod(light_ssao, vec3(cache.uv, 0.0), 0.0).r;
}
#else
return 1.0;
#endif
}
float SpecularAO_Lagarde(float NoV, float visibility, float roughness) {
return saturate(pow(NoV + visibility, exp2(-16.0 * roughness - 1.0)) - 1.0 + visibility);
}
float sphericalCapsIntersection(float cosCap1, float cosCap2, float cosDistance) {
float r1 = acosFastPositive(cosCap1);
float r2 = acosFastPositive(cosCap2);
float d  = acosFast(cosDistance);
if (min(r1, r2) <= max(r1, r2) - d) {
return 1.0 - max(cosCap1, cosCap2);
} else if (r1 + r2 <= d) {
return 0.0;
}
float delta = abs(r1 - r2);
float x = 1.0 - saturate((d - delta) / max(r1 + r2 - delta, 1e-4));
float area = sq(x) * (-2.0 * x + 3.0);
return area * (1.0 - max(cosCap1, cosCap2));
}
float SpecularAO_Cones(vec3 bentNormal, float visibility, float roughness) {
float cosAv = sqrt(1.0 - visibility);
float cosAs = exp2(-3.321928 * sq(roughness));
float cosB  = dot(bentNormal, shading_reflected);
float ao = sphericalCapsIntersection(cosAv, cosAs, cosB) / (1.0 - cosAs);
return mix(1.0, ao, smoothstep(0.01, 0.09, roughness));
}
vec3 unpackBentNormal(vec3 bn) {
return bn * 2.0 - 1.0;
}
float specularAO(float NoV, float visibility, float roughness, const in SSAOInterpolationCache cache) {
float specularAO = 1.0;
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED)
#if SPECULAR_AMBIENT_OCCLUSION == SPECULAR_AO_SIMPLE
specularAO = SpecularAO_Lagarde(NoV, visibility, roughness);
#elif SPECULAR_AMBIENT_OCCLUSION == SPECULAR_AO_BENT_NORMALS
#   if defined(MATERIAL_HAS_BENT_NORMAL)
specularAO = SpecularAO_Cones(shading_bentNormal, visibility, roughness);
#   else
specularAO = SpecularAO_Cones(shading_normal, visibility, roughness);
#   endif
#endif
if (frameUniforms.aoBentNormals > 0.0) {
vec3 bn;
if (frameUniforms.aoSamplingQualityAndEdgeDistance > 0.0) {
#if defined(FILAMENT_HAS_FEATURE_TEXTURE_GATHER)
vec4 bnr = textureGather(light_ssao, vec3(cache.uv, 1.0), 0);
vec4 bng = textureGather(light_ssao, vec3(cache.uv, 1.0), 1);
vec4 bnb = textureGather(light_ssao, vec3(cache.uv, 1.0), 2);
#else
vec3 s01 = textureLodOffset(light_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(0, 1)).rgb;
vec3 s11 = textureLodOffset(light_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(1, 1)).rgb;
vec3 s10 = textureLodOffset(light_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(1, 0)).rgb;
vec3 s00 = textureLodOffset(light_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(0, 0)).rgb;
vec4 bnr = vec4(s01.r, s11.r, s10.r, s00.r);
vec4 bng = vec4(s01.g, s11.g, s10.g, s00.g);
vec4 bnb = vec4(s01.b, s11.b, s10.b, s00.b);
#endif
bn.r = dot(bnr, cache.weights);
bn.g = dot(bng, cache.weights);
bn.b = dot(bnb, cache.weights);
} else {
bn = textureLod(light_ssao, vec3(cache.uv, 1.0), 0.0).xyz;
}
bn = unpackBentNormal(bn);
bn = normalize(bn);
float ssSpecularAO = SpecularAO_Cones(bn, visibility, roughness);
specularAO = min(specularAO, ssSpecularAO);
}
#endif
return specularAO;
}
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1
vec3 gtaoMultiBounce(float visibility, const vec3 albedo) {
vec3 a =  2.0404 * albedo - 0.3324;
vec3 b = -4.7951 * albedo + 0.6417;
vec3 c =  2.7552 * albedo + 0.6903;
return max(vec3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}
#endif
void multiBounceAO(float visibility, const vec3 albedo, inout vec3 color) {
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1
color *= gtaoMultiBounce(visibility, albedo);
#endif
}
void multiBounceSpecularAO(float visibility, const vec3 albedo, inout vec3 color) {
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1 && SPECULAR_AMBIENT_OCCLUSION != SPECULAR_AO_OFF
color *= gtaoMultiBounce(visibility, albedo);
#endif
}
float singleBounceAO(float visibility) {
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1
return 1.0;
#else
return visibility;
#endif
}

#define SPHERICAL_HARMONICS_BANDS           3
#define IBL_INTEGRATION_PREFILTERED_CUBEMAP         0
#define IBL_INTEGRATION_IMPORTANCE_SAMPLING         1
#define IBL_INTEGRATION                             IBL_INTEGRATION_PREFILTERED_CUBEMAP
#define IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT   64
vec3 decodeDataForIBL(const vec4 data) {
return data.rgb;
}
vec3 PrefilteredDFG_LUT(float lod, float NoV) {
return textureLod(light_iblDFG, vec2(NoV, lod), 0.0).rgb;
}
vec3 prefilteredDFG(float perceptualRoughness, float NoV) {
return PrefilteredDFG_LUT(perceptualRoughness, NoV);
}
vec3 Irradiance_SphericalHarmonics(const vec3 n) {
return max(
frameUniforms.iblSH[0]
#if SPHERICAL_HARMONICS_BANDS >= 2
+ frameUniforms.iblSH[1] * (n.y)
+ frameUniforms.iblSH[2] * (n.z)
+ frameUniforms.iblSH[3] * (n.x)
#endif
#if SPHERICAL_HARMONICS_BANDS >= 3
+ frameUniforms.iblSH[4] * (n.y * n.x)
+ frameUniforms.iblSH[5] * (n.y * n.z)
+ frameUniforms.iblSH[6] * (3.0 * n.z * n.z - 1.0)
+ frameUniforms.iblSH[7] * (n.z * n.x)
+ frameUniforms.iblSH[8] * (n.x * n.x - n.y * n.y)
#endif
, 0.0);
}
vec3 Irradiance_RoughnessOne(const vec3 n) {
return decodeDataForIBL(textureLod(light_iblSpecular, n, frameUniforms.iblRoughnessOneLevel));
}
vec3 diffuseIrradiance(const vec3 n) {
if (frameUniforms.iblSH[0].x == 65504.0) {
#if FILAMENT_QUALITY < FILAMENT_QUALITY_HIGH
return Irradiance_RoughnessOne(n);
#else
ivec2 s = textureSize(light_iblSpecular, int(frameUniforms.iblRoughnessOneLevel));
float du = 1.0 / float(s.x);
float dv = 1.0 / float(s.y);
vec3 m0 = normalize(cross(n, vec3(0.0, 1.0, 0.0)));
vec3 m1 = cross(m0, n);
vec3 m0du = m0 * du;
vec3 m1dv = m1 * dv;
vec3 c;
c  = Irradiance_RoughnessOne(n - m0du - m1dv);
c += Irradiance_RoughnessOne(n + m0du - m1dv);
c += Irradiance_RoughnessOne(n + m0du + m1dv);
c += Irradiance_RoughnessOne(n - m0du + m1dv);
return c * 0.25;
#endif
} else {
return Irradiance_SphericalHarmonics(n);
}
}
float perceptualRoughnessToLod(float perceptualRoughness) {
return frameUniforms.iblRoughnessOneLevel * perceptualRoughness * (2.0 - perceptualRoughness);
}
vec3 prefilteredRadiance(const vec3 r, float perceptualRoughness) {
float lod = perceptualRoughnessToLod(perceptualRoughness);
return decodeDataForIBL(textureLod(light_iblSpecular, r, lod));
}
vec3 prefilteredRadiance(const vec3 r, float roughness, float offset) {
float lod = frameUniforms.iblRoughnessOneLevel * roughness;
return decodeDataForIBL(textureLod(light_iblSpecular, r, lod + offset));
}
vec3 getSpecularDominantDirection(const vec3 n, const vec3 r, float roughness) {
return mix(r, n, roughness * roughness);
}
vec3 specularDFG(const PixelParams pixel) {
#if defined(SHADING_MODEL_CLOTH)
return pixel.f0 * pixel.dfg.z;
#else
return mix(pixel.dfg.xxx, pixel.dfg.yyy, pixel.f0);
#endif
}
vec3 getReflectedVector(const PixelParams pixel, const vec3 n) {
#if defined(MATERIAL_HAS_ANISOTROPY)
vec3 r = getReflectedVector(pixel, shading_view, n);
#else
vec3 r = shading_reflected;
#endif
return getSpecularDominantDirection(n, r, pixel.roughness);
}
#if IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
vec2 hammersley(uint index) {
const uint numSamples = uint(IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT);
const float invNumSamples = 1.0 / float(numSamples);
const float tof = 0.5 / float(0x80000000U);
uint bits = index;
bits = (bits << 16u) | (bits >> 16u);
bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
return vec2(float(index) * invNumSamples, float(bits) * tof);
}
vec3 importanceSamplingNdfDggx(vec2 u, float roughness) {
float a2 = roughness * roughness;
float phi = 2.0 * PI * u.x;
float cosTheta2 = (1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y);
float cosTheta = sqrt(cosTheta2);
float sinTheta = sqrt(1.0 - cosTheta2);
return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
vec3 hemisphereCosSample(vec2 u) {
float phi = 2.0f * PI * u.x;
float cosTheta2 = 1.0 - u.y;
float cosTheta = sqrt(cosTheta2);
float sinTheta = sqrt(1.0 - cosTheta2);
return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}
vec3 importanceSamplingVNdfDggx(vec2 u, float roughness, vec3 v) {
float alpha = roughness;
v = normalize(vec3(alpha * v.x, alpha * v.y, v.z));
vec3 up = abs(v.z) < 0.9999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
vec3 t = normalize(cross(up, v));
vec3 b = cross(t, v);
float a = 1.0 / (1.0 + v.z);
float r = sqrt(u.x);
float phi = (u.y < a) ? u.y / a * PI : PI + (u.y - a) / (1.0 - a) * PI;
float p1 = r * cos(phi);
float p2 = r * sin(phi) * ((u.y < a) ? 1.0 : v.z);
vec3 h = p1 * t + p2 * b + sqrt(max(0.0, 1.0 - p1*p1 - p2*p2)) * v;
h = normalize(vec3(alpha * h.x, alpha * h.y, max(0.0, h.z)));
return h;
}
float prefilteredImportanceSampling(float ipdf, float omegaP) {
const float numSamples = float(IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT);
const float invNumSamples = 1.0 / float(numSamples);
const float K = 4.0;
float omegaS = invNumSamples * ipdf;
float mipLevel = log2(K * omegaS / omegaP) * 0.5;
return mipLevel;
}
vec3 isEvaluateSpecularIBL(const PixelParams pixel, const vec3 n, const vec3 v, const float NoV) {
const uint numSamples = uint(IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT);
const float invNumSamples = 1.0 / float(numSamples);
const vec3 up = vec3(0.0, 0.0, 1.0);
mat3 T;
T[0] = normalize(cross(up, n));
T[1] = cross(n, T[0]);
T[2] = n;
const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
float a = 2.0 * PI * fract(m.z * fract(dot(gl_FragCoord.xy, m.xy)));
float c = cos(a);
float s = sin(a);
mat3 R;
R[0] = vec3( c, s, 0);
R[1] = vec3(-s, c, 0);
R[2] = vec3( 0, 0, 1);
T *= R;
float roughness = pixel.roughness;
float dim = float(textureSize(light_iblSpecular, 0).x);
float omegaP = (4.0 * PI) / (6.0 * dim * dim);
vec3 indirectSpecular = vec3(0.0);
for (uint i = 0u; i < numSamples; i++) {
vec2 u = hammersley(i);
vec3 h = T * importanceSamplingNdfDggx(u, roughness);
vec3 l = getReflectedVector(pixel, v, h);
float NoL = saturate(dot(n, l));
if (NoL > 0.0) {
float NoH = dot(n, h);
float LoH = saturate(dot(l, h));
float ipdf = (4.0 * LoH) / (D_GGX(roughness, NoH, h) * NoH);
float mipLevel = prefilteredImportanceSampling(ipdf, omegaP);
vec3 L = decodeDataForIBL(textureLod(light_iblSpecular, l, mipLevel));
float D = distribution(roughness, NoH, h);
float V = visibility(roughness, NoV, NoL);
vec3 F = fresnel(pixel.f0, LoH);
vec3 Fr = F * (D * V * NoL * ipdf * invNumSamples);
indirectSpecular += (Fr * L);
}
}
return indirectSpecular;
}
vec3 isEvaluateDiffuseIBL(const PixelParams pixel, vec3 n, vec3 v) {
const uint numSamples = uint(IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT);
const float invNumSamples = 1.0 / float(numSamples);
const vec3 up = vec3(0.0, 0.0, 1.0);
mat3 T;
T[0] = normalize(cross(up, n));
T[1] = cross(n, T[0]);
T[2] = n;
const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
float a = 2.0 * PI * fract(m.z * fract(dot(gl_FragCoord.xy, m.xy)));
float c = cos(a);
float s = sin(a);
mat3 R;
R[0] = vec3( c, s, 0);
R[1] = vec3(-s, c, 0);
R[2] = vec3( 0, 0, 1);
T *= R;
float dim = float(textureSize(light_iblSpecular, 0).x);
float omegaP = (4.0 * PI) / (6.0 * dim * dim);
vec3 indirectDiffuse = vec3(0.0);
for (uint i = 0u; i < numSamples; i++) {
vec2 u = hammersley(i);
vec3 h = T * hemisphereCosSample(u);
vec3 l = getReflectedVector(pixel, v, h);
float NoL = saturate(dot(n, l));
if (NoL > 0.0) {
float ipdf = PI / NoL;
float mipLevel = prefilteredImportanceSampling(ipdf, omegaP) + 1.0;
vec3 L = decodeDataForIBL(textureLod(light_iblSpecular, l, mipLevel));
indirectDiffuse += L;
}
}
return indirectDiffuse * invNumSamples;
}
void isEvaluateClearCoatIBL(const PixelParams pixel, float specularAO, inout vec3 Fd, inout vec3 Fr) {
#if defined(MATERIAL_HAS_CLEAR_COAT)
#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
float clearCoatNoV = clampNoV(dot(shading_clearCoatNormal, shading_view));
vec3 clearCoatNormal = shading_clearCoatNormal;
#else
float clearCoatNoV = shading_NoV;
vec3 clearCoatNormal = shading_normal;
#endif
float Fc = F_Schlick(0.04, 1.0, clearCoatNoV) * pixel.clearCoat;
float attenuation = 1.0 - Fc;
Fd *= attenuation;
Fr *= attenuation;
PixelParams p;
p.perceptualRoughness = pixel.clearCoatPerceptualRoughness;
p.f0 = vec3(0.04);
p.roughness = perceptualRoughnessToRoughness(p.perceptualRoughness);
#if defined(MATERIAL_HAS_ANISOTROPY)
p.anisotropy = 0.0;
#endif
vec3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, shading_view, clearCoatNoV);
Fr += clearCoatLobe * (specularAO * pixel.clearCoat);
#endif
}
#endif
void evaluateClothIndirectDiffuseBRDF(const PixelParams pixel, inout float diffuse) {
#if defined(SHADING_MODEL_CLOTH)
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
diffuse *= Fd_Wrap(shading_NoV, 0.5);
#endif
#endif
}
void evaluateSheenIBL(const PixelParams pixel, float diffuseAO,
const in SSAOInterpolationCache cache, inout vec3 Fd, inout vec3 Fr) {
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
#if defined(MATERIAL_HAS_SHEEN_COLOR)
Fd *= pixel.sheenScaling;
Fr *= pixel.sheenScaling;
vec3 reflectance = pixel.sheenDFG * pixel.sheenColor;
reflectance *= specularAO(shading_NoV, diffuseAO, pixel.sheenRoughness, cache);
Fr += reflectance * prefilteredRadiance(shading_reflected, pixel.sheenPerceptualRoughness);
#endif
#endif
}
void evaluateClearCoatIBL(const PixelParams pixel, float diffuseAO,
const in SSAOInterpolationCache cache, inout vec3 Fd, inout vec3 Fr) {
#if IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
float specularAO = specularAO(shading_NoV, diffuseAO, pixel.clearCoatRoughness, cache);
isEvaluateClearCoatIBL(pixel, specularAO, Fd, Fr);
return;
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT)
if (pixel.clearCoat > 0.0) {
#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
float clearCoatNoV = clampNoV(dot(shading_clearCoatNormal, shading_view));
vec3 clearCoatR = reflect(-shading_view, shading_clearCoatNormal);
#else
float clearCoatNoV = shading_NoV;
vec3 clearCoatR = shading_reflected;
#endif
float Fc = F_Schlick(0.04, 1.0, clearCoatNoV) * pixel.clearCoat;
float attenuation = 1.0 - Fc;
Fd *= attenuation;
Fr *= attenuation;
float specularAO = specularAO(clearCoatNoV, diffuseAO, pixel.clearCoatRoughness, cache);
Fr += prefilteredRadiance(clearCoatR, pixel.clearCoatPerceptualRoughness) * (specularAO * Fc);
}
#endif
}
void evaluateSubsurfaceIBL(const PixelParams pixel, const vec3 diffuseIrradiance,
inout vec3 Fd, inout vec3 Fr) {
#if defined(SHADING_MODEL_SUBSURFACE)
vec3 viewIndependent = diffuseIrradiance;
vec3 viewDependent = prefilteredRadiance(-shading_view, pixel.roughness, 1.0 + pixel.thickness);
float attenuation = (1.0 - pixel.thickness) / (2.0 * PI);
Fd += pixel.subsurfaceColor * (viewIndependent + viewDependent) * attenuation;
#elif defined(SHADING_MODEL_CLOTH) && defined(MATERIAL_HAS_SUBSURFACE_COLOR)
Fd *= saturate(pixel.subsurfaceColor + shading_NoV);
#endif
}
#if defined(MATERIAL_HAS_REFRACTION)
struct Refraction {
vec3 position;
vec3 direction;
float d;
};
void refractionSolidSphere(const PixelParams pixel,
const vec3 n, vec3 r, out Refraction ray) {
r = refract(r, n, pixel.etaIR);
float NoR = dot(n, r);
float d = pixel.thickness * -NoR;
ray.position = vec3(shading_position + r * d);
ray.d = d;
vec3 n1 = normalize(NoR * r - n * 0.5);
ray.direction = refract(r, n1,  pixel.etaRI);
}
void refractionSolidBox(const PixelParams pixel,
const vec3 n, vec3 r, out Refraction ray) {
vec3 rr = refract(r, n, pixel.etaIR);
float NoR = dot(n, rr);
float d = pixel.thickness / max(-NoR, 0.001);
ray.position = vec3(shading_position + rr * d);
ray.direction = r;
ray.d = d;
#if REFRACTION_MODE == REFRACTION_MODE_CUBEMAP
float envDistance = 10.0;
ray.direction = normalize((ray.position - shading_position) + ray.direction * envDistance);
#endif
}
void refractionThinSphere(const PixelParams pixel,
const vec3 n, vec3 r, out Refraction ray) {
float d = 0.0;
#if defined(MATERIAL_HAS_MICRO_THICKNESS)
vec3 rr = refract(r, n, pixel.etaIR);
float NoR = dot(n, rr);
d = pixel.uThickness / max(-NoR, 0.001);
ray.position = vec3(shading_position + rr * d);
#else
ray.position = vec3(shading_position);
#endif
ray.direction = r;
ray.d = d;
}
vec3 evaluateRefraction(
const PixelParams pixel,
const vec3 n0, vec3 E) {
Refraction ray;
#if REFRACTION_TYPE == REFRACTION_TYPE_SOLID
refractionSolidSphere(pixel, n0, -shading_view, ray);
#elif REFRACTION_TYPE == REFRACTION_TYPE_THIN
refractionThinSphere(pixel, n0, -shading_view, ray);
#else
#error invalid REFRACTION_TYPE
#endif
#if defined(MATERIAL_HAS_ABSORPTION)
#if defined(MATERIAL_HAS_THICKNESS) || defined(MATERIAL_HAS_MICRO_THICKNESS)
vec3 T = min(vec3(1.0), exp(-pixel.absorption * ray.d));
#else
vec3 T = 1.0 - pixel.absorption;
#endif
#endif
float perceptualRoughness = mix(pixel.perceptualRoughnessUnclamped, 0.0,
saturate(pixel.etaIR * 3.0 - 2.0));
#if REFRACTION_TYPE == REFRACTION_TYPE_THIN
E *= 1.0 + pixel.transmission * (1.0 - E.g) / (1.0 + E.g);
#endif
#if REFRACTION_MODE == REFRACTION_MODE_CUBEMAP
vec3 Ft = prefilteredRadiance(ray.direction, perceptualRoughness) * frameUniforms.iblLuminance;
#else
vec3 Ft;
vec4 p = vec4(getClipFromWorldMatrix() * vec4(ray.position, 1.0));
p.xy = uvToRenderTargetUV(p.xy * (0.5 / p.w) + 0.5);
const float invLog2sqrt5 = 0.8614;
float lod = max(0.0, (2.0f * log2(perceptualRoughness) + frameUniforms.refractionLodOffset) * invLog2sqrt5);
Ft = textureLod(light_ssr, vec3(p.xy, 0.0), lod).rgb;
#endif
Ft *= pixel.diffuseColor;
Ft *= 1.0 - E;
#if defined(MATERIAL_HAS_ABSORPTION)
Ft *= T;
#endif
return Ft;
}
#endif
void evaluateIBL(const MaterialInputs material, const PixelParams pixel, inout vec3 color) {
vec3 Fr = vec3(0.0f);
SSAOInterpolationCache interpolationCache;
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED) || defined(MATERIAL_HAS_REFLECTIONS)
interpolationCache.uv = uvToRenderTargetUV(getNormalizedPhysicalViewportCoord().xy);
#endif
#if defined(MATERIAL_HAS_REFLECTIONS)
vec4 ssrFr = vec4(0.0f);
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED)
if (frameUniforms.ssrDistance > 0.0f) {
const float maxPerceptualRoughness = sqrt(0.5);
if (pixel.perceptualRoughness < maxPerceptualRoughness) {
const float invLog2sqrt5 = 0.8614;
float d = -mulMat4x4Float3(getViewFromWorldMatrix(), getWorldPosition()).z;
float lod = max(0.0, (log2(pixel.roughness / d) + frameUniforms.refractionLodOffset) * invLog2sqrt5);
ssrFr = textureLod(light_ssr, vec3(interpolationCache.uv, 1.0), lod);
}
}
#else
#endif
#else
const vec4 ssrFr = vec4(0.0f);
#endif
#if IBL_INTEGRATION == IBL_INTEGRATION_PREFILTERED_CUBEMAP
vec3 E = specularDFG(pixel);
if (ssrFr.a < 1.0f) {
vec3 r = getReflectedVector(pixel, shading_normal);
Fr = E * prefilteredRadiance(r, pixel.perceptualRoughness);
}
#elif IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
vec3 E = vec3(0.0);
if (ssrFr.a < 1.0f) {
Fr = isEvaluateSpecularIBL(pixel, shading_normal, shading_view, shading_NoV);
}
#endif
float ssao = evaluateSSAO(interpolationCache);
float diffuseAO = min(material.ambientOcclusion, ssao);
float specularAO = specularAO(shading_NoV, diffuseAO, pixel.roughness, interpolationCache);
vec3 specularSingleBounceAO = singleBounceAO(specularAO) * pixel.energyCompensation;
Fr *= specularSingleBounceAO;
#if defined(MATERIAL_HAS_REFLECTIONS)
ssrFr.rgb *= specularSingleBounceAO;
#endif
float diffuseBRDF = singleBounceAO(diffuseAO);
evaluateClothIndirectDiffuseBRDF(pixel, diffuseBRDF);
#if defined(MATERIAL_HAS_BENT_NORMAL)
vec3 diffuseNormal = shading_bentNormal;
#else
vec3 diffuseNormal = shading_normal;
#endif
#if IBL_INTEGRATION == IBL_INTEGRATION_PREFILTERED_CUBEMAP
vec3 diffuseIrradiance = diffuseIrradiance(diffuseNormal);
#elif IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
vec3 diffuseIrradiance = isEvaluateDiffuseIBL(pixel, diffuseNormal, shading_view);
#endif
vec3 Fd = pixel.diffuseColor * diffuseIrradiance * (1.0 - E) * diffuseBRDF;
evaluateSubsurfaceIBL(pixel, diffuseIrradiance, Fd, Fr);
multiBounceAO(diffuseAO, pixel.diffuseColor, Fd);
multiBounceSpecularAO(specularAO, pixel.f0, Fr);
evaluateSheenIBL(pixel, diffuseAO, interpolationCache, Fd, Fr);
evaluateClearCoatIBL(pixel, diffuseAO, interpolationCache, Fd, Fr);
Fr *= frameUniforms.iblLuminance;
Fd *= frameUniforms.iblLuminance;
#if defined(MATERIAL_HAS_REFRACTION)
vec3 Ft = evaluateRefraction(pixel, shading_normal, E);
Ft *= pixel.transmission;
Fd *= (1.0 - pixel.transmission);
#endif
#if defined(MATERIAL_HAS_REFLECTIONS)
Fr = Fr * (1.0 - ssrFr.a) + (E * ssrFr.rgb);
#endif
color.rgb += Fr + Fd;
#if defined(MATERIAL_HAS_REFRACTION)
color.rgb += Ft;
#endif
}

#if FILAMENT_QUALITY < FILAMENT_QUALITY_HIGH
#define SUN_AS_AREA_LIGHT
#endif
vec3 sampleSunAreaLight(const vec3 lightDirection) {
#if defined(SUN_AS_AREA_LIGHT)
if (frameUniforms.sun.w >= 0.0) {
float LoR = dot(lightDirection, shading_reflected);
float d = frameUniforms.sun.x;
highp vec3 s = shading_reflected - LoR * lightDirection;
return LoR < d ?
normalize(lightDirection * d + normalize(s) * frameUniforms.sun.y) : shading_reflected;
}
#endif
return lightDirection;
}
Light getDirectionalLight() {
Light light;
light.colorIntensity = frameUniforms.lightColorIntensity;
light.l = sampleSunAreaLight(frameUniforms.lightDirection);
light.attenuation = 1.0;
light.NoL = saturate(dot(shading_normal, light.l));
light.channels = frameUniforms.lightChannels & 0xFFu;
return light;
}
void evaluateDirectionalLight(const MaterialInputs material,
const PixelParams pixel, inout vec3 color) {
Light light = getDirectionalLight();
uint channels = object_uniforms.flagsChannels & 0xFFu;
if ((light.channels & channels) == 0u) {
return;
}
#if defined(MATERIAL_CAN_SKIP_LIGHTING)
if (light.NoL <= 0.0) {
return;
}
#endif
float visibility = 1.0;
#if defined(VARIANT_HAS_SHADOWING)
if (light.NoL > 0.0) {
float ssContactShadowOcclusion = 0.0;
uint cascade = getShadowCascade();
bool cascadeHasVisibleShadows = bool(frameUniforms.cascades & ((1u << cascade) << 8u));
bool hasDirectionalShadows = bool(frameUniforms.directionalShadows & 1u);
if (hasDirectionalShadows && cascadeHasVisibleShadows) {
highp vec4 shadowPosition = getShadowPosition(cascade);
visibility = shadow(true, light_shadowMap, cascade, shadowPosition, 0.0f);
}
if ((frameUniforms.directionalShadows & 0x2u) != 0u && visibility > 0.0) {
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_CONTACT_SHADOWS_BIT) != 0u) {
ssContactShadowOcclusion = screenSpaceContactShadow(light.l);
}
}
visibility *= 1.0 - ssContactShadowOcclusion;
#if defined(MATERIAL_HAS_AMBIENT_OCCLUSION)
visibility *= computeMicroShadowing(light.NoL, material.ambientOcclusion);
#endif
#if defined(MATERIAL_CAN_SKIP_LIGHTING)
if (visibility <= 0.0) {
return;
}
#endif
}
#endif
#if defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
color.rgb += customSurfaceShading(material, pixel, light, visibility);
#else
color.rgb += surfaceShading(pixel, light, visibility);
#endif
}

#define FROXEL_BUFFER_WIDTH_SHIFT   6u
#define FROXEL_BUFFER_WIDTH         (1u << FROXEL_BUFFER_WIDTH_SHIFT)
#define FROXEL_BUFFER_WIDTH_MASK    (FROXEL_BUFFER_WIDTH - 1u)
#define LIGHT_TYPE_POINT            0u
#define LIGHT_TYPE_SPOT             1u
struct FroxelParams {
uint recordOffset;
uint count;
};
uvec3 getFroxelCoords(const highp vec3 fragCoords) {
uvec3 froxelCoord;
froxelCoord.xy = uvec2(fragCoords.xy * frameUniforms.froxelCountXY);
highp float viewSpaceNormalizedZ = frameUniforms.zParams.x * fragCoords.z + frameUniforms.zParams.y;
float zSliceCount = frameUniforms.zParams.w;
float sliceZWithoutOffset = log2(viewSpaceNormalizedZ) * frameUniforms.zParams.z;
froxelCoord.z = uint(clamp(sliceZWithoutOffset + zSliceCount, 0.0, zSliceCount - 1.0));
return froxelCoord;
}
uint getFroxelIndex(const highp vec3 fragCoords) {
uvec3 froxelCoord = getFroxelCoords(fragCoords);
return froxelCoord.x * frameUniforms.fParams.x +
froxelCoord.y * frameUniforms.fParams.y +
froxelCoord.z * frameUniforms.fParams.z;
}
ivec2 getFroxelTexCoord(uint froxelIndex) {
return ivec2(froxelIndex & FROXEL_BUFFER_WIDTH_MASK, froxelIndex >> FROXEL_BUFFER_WIDTH_SHIFT);
}
FroxelParams getFroxelParams(uint froxelIndex) {
ivec2 texCoord = getFroxelTexCoord(froxelIndex);
uvec2 entry = texelFetch(light_froxels, texCoord, 0).rg;
FroxelParams froxel;
froxel.recordOffset = entry.r;
froxel.count = entry.g & 0xFFu;
return froxel;
}
uint getLightIndex(const uint index) {
uint v = index >> 4u;
uint c = (index >> 2u) & 0x3u;
uint s = (index & 0x3u) * 8u;
highp uvec4 d = froxelRecordUniforms.records[v];
return (d[c] >> s) & 0xFFu;
}
float getSquareFalloffAttenuation(float distanceSquare, float falloff) {
float factor = distanceSquare * falloff;
float smoothFactor = saturate(1.0 - factor * factor);
return smoothFactor * smoothFactor;
}
float getDistanceAttenuation(const highp vec3 posToLight, float falloff) {
float distanceSquare = dot(posToLight, posToLight);
float attenuation = getSquareFalloffAttenuation(distanceSquare, falloff);
highp vec3 v = getWorldPosition() - getWorldCameraPosition();
float d = dot(v, v);
attenuation *= saturate(frameUniforms.lightFarAttenuationParams.x - d * frameUniforms.lightFarAttenuationParams.y);
return attenuation / max(distanceSquare, 1e-4);
}
float getAngleAttenuation(const highp vec3 lightDir, const highp vec3 l, const highp vec2 scaleOffset) {
float cd = dot(lightDir, l);
float attenuation = saturate(cd * scaleOffset.x + scaleOffset.y);
return attenuation * attenuation;
}
Light getLight(const uint lightIndex) {
highp mat4 data = lightsUniforms.lights[lightIndex];
highp vec4 positionFalloff = data[0];
highp vec3 direction = data[1].xyz;
vec4 colorIES = vec4(
unpackHalf2x16(floatBitsToUint(data[2][0])),
unpackHalf2x16(floatBitsToUint(data[2][1]))
);
highp vec2 scaleOffset = data[2].zw;
highp float intensity = data[3][1];
highp uint typeShadow = floatBitsToUint(data[3][2]);
highp uint channels = floatBitsToUint(data[3][3]);
highp vec3 worldPosition = getWorldPosition();
highp vec3 posToLight = positionFalloff.xyz - worldPosition;
Light light;
light.colorIntensity.rgb = colorIES.rgb;
light.colorIntensity.w = computePreExposedIntensity(intensity, frameUniforms.exposure);
light.l = normalize(posToLight);
light.attenuation = getDistanceAttenuation(posToLight, positionFalloff.w);
light.direction = direction;
light.NoL = saturate(dot(shading_normal, light.l));
light.worldPosition = positionFalloff.xyz;
light.channels = channels;
light.contactShadows = bool(typeShadow & 0x10u);
#if defined(VARIANT_HAS_DYNAMIC_LIGHTING)
light.type = (typeShadow & 0x1u);
#if defined(VARIANT_HAS_SHADOWING)
light.shadowIndex = (typeShadow >>  8u) & 0xFFu;
light.castsShadows   = bool(channels & 0x10000u);
if (light.type == LIGHT_TYPE_SPOT) {
light.zLight = dot(shadowUniforms.shadows[light.shadowIndex].lightFromWorldZ, vec4(worldPosition, 1.0));
}
#endif
if (light.type == LIGHT_TYPE_SPOT) {
light.attenuation *= getAngleAttenuation(-direction, light.l, scaleOffset);
}
#endif
return light;
}
void evaluatePunctualLights(const MaterialInputs material,
const PixelParams pixel, inout vec3 color) {
FroxelParams froxel = getFroxelParams(getFroxelIndex(getNormalizedPhysicalViewportCoord()));
uint index = froxel.recordOffset;
uint end = index + froxel.count;
uint channels = object_uniforms.flagsChannels & 0xFFu;
for ( ; index < end; index++) {
uint lightIndex = getLightIndex(index);
Light light = getLight(lightIndex);
if ((light.channels & channels) == 0u) {
continue;
}
#if defined(MATERIAL_CAN_SKIP_LIGHTING)
if (light.NoL <= 0.0 || light.attenuation <= 0.0) {
continue;
}
#endif
float visibility = 1.0;
#if defined(VARIANT_HAS_SHADOWING)
if (light.NoL > 0.0) {
if (light.castsShadows) {
uint shadowIndex = light.shadowIndex;
if (light.type == LIGHT_TYPE_POINT) {
highp vec3 r = getWorldPosition() - light.worldPosition;
uint face = getPointLightFace(r);
shadowIndex += face;
light.zLight = dot(shadowUniforms.shadows[shadowIndex].lightFromWorldZ,
vec4(getWorldPosition(), 1.0));
}
highp vec4 shadowPosition = getShadowPosition(shadowIndex, light.direction, light.zLight);
visibility = shadow(false, light_shadowMap, shadowIndex,
shadowPosition, light.zLight);
}
if (light.contactShadows && visibility > 0.0) {
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_CONTACT_SHADOWS_BIT) != 0u) {
visibility *= 1.0 - screenSpaceContactShadow(light.l);
}
}
#if defined(MATERIAL_CAN_SKIP_LIGHTING)
if (visibility <= 0.0) {
continue;
}
#endif
}
#endif
#if defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
color.rgb += customSurfaceShading(material, pixel, light, visibility);
#else
color.rgb += surfaceShading(pixel, light, visibility);
#endif
}
}

#if defined(BLEND_MODE_MASKED)
float computeMaskedAlpha(float a) {
return (a - getMaskThreshold()) / max(fwidth(a), 1e-3) + 0.5;
}
float computeDiffuseAlpha(float a) {
return (frameUniforms.needsAlphaChannel == 1.0) ? 1.0 : a;
}
void applyAlphaMask(inout vec4 baseColor) {
baseColor.a = computeMaskedAlpha(baseColor.a);
if (baseColor.a <= 0.0) {
discard;
}
}
#else
float computeDiffuseAlpha(float a) {
#if defined(BLEND_MODE_TRANSPARENT) || defined(BLEND_MODE_FADE)
return a;
#else
return 1.0;
#endif
}
void applyAlphaMask(inout vec4 baseColor) {}
#endif
#if defined(GEOMETRIC_SPECULAR_AA)
float normalFiltering(float perceptualRoughness, const vec3 worldNormal) {
vec3 du = dFdx(worldNormal);
vec3 dv = dFdy(worldNormal);
float variance = materialParams._specularAntiAliasingVariance * (dot(du, du) + dot(dv, dv));
float roughness = perceptualRoughnessToRoughness(perceptualRoughness);
float kernelRoughness = min(2.0 * variance, materialParams._specularAntiAliasingThreshold);
float squareRoughness = saturate(roughness * roughness + kernelRoughness);
return roughnessToPerceptualRoughness(sqrt(squareRoughness));
}
#endif
void getCommonPixelParams(const MaterialInputs material, inout PixelParams pixel) {
vec4 baseColor = material.baseColor;
applyAlphaMask(baseColor);
#if defined(BLEND_MODE_FADE) && !defined(SHADING_MODEL_UNLIT)
unpremultiply(baseColor);
#endif
#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
vec3 specularColor = material.specularColor;
float metallic = computeMetallicFromSpecularColor(specularColor);
pixel.diffuseColor = computeDiffuseColor(baseColor, metallic);
pixel.f0 = specularColor;
#elif !defined(SHADING_MODEL_CLOTH)
pixel.diffuseColor = computeDiffuseColor(baseColor, material.metallic);
#if !defined(SHADING_MODEL_SUBSURFACE) && (!defined(MATERIAL_HAS_REFLECTANCE) && defined(MATERIAL_HAS_IOR))
float reflectance = iorToF0(max(1.0, material.ior), 1.0);
#else
float reflectance = computeDielectricF0(material.reflectance);
#endif
pixel.f0 = computeF0(baseColor, material.metallic, reflectance);
#else
pixel.diffuseColor = baseColor.rgb;
pixel.f0 = material.sheenColor;
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
pixel.subsurfaceColor = material.subsurfaceColor;
#endif
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
#if defined(MATERIAL_HAS_REFRACTION)
const float airIor = 1.0;
#if !defined(MATERIAL_HAS_IOR)
float materialor = f0ToIor(pixel.f0.g);
#else
float materialor = max(1.0, material.ior);
#endif
pixel.etaIR = airIor / materialor;
pixel.etaRI = materialor / airIor;
#if defined(MATERIAL_HAS_TRANSMISSION)
pixel.transmission = saturate(material.transmission);
#else
pixel.transmission = 1.0;
#endif
#if defined(MATERIAL_HAS_ABSORPTION)
#if defined(MATERIAL_HAS_THICKNESS) || defined(MATERIAL_HAS_MICRO_THICKNESS)
pixel.absorption = max(vec3(0.0), material.absorption);
#else
pixel.absorption = saturate(material.absorption);
#endif
#else
pixel.absorption = vec3(0.0);
#endif
#if defined(MATERIAL_HAS_THICKNESS)
pixel.thickness = max(0.0, material.thickness);
#endif
#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
pixel.uThickness = max(0.0, material.microThickness);
#else
pixel.uThickness = 0.0;
#endif
#endif
#endif
}
void getSheenPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_SHEEN_COLOR) && !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
pixel.sheenColor = material.sheenColor;
float sheenPerceptualRoughness = material.sheenRoughness;
sheenPerceptualRoughness = clamp(sheenPerceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
#if defined(GEOMETRIC_SPECULAR_AA)
sheenPerceptualRoughness =
normalFiltering(sheenPerceptualRoughness, getWorldGeometricNormalVector());
#endif
pixel.sheenPerceptualRoughness = sheenPerceptualRoughness;
pixel.sheenRoughness = perceptualRoughnessToRoughness(sheenPerceptualRoughness);
#endif
}
void getClearCoatPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_CLEAR_COAT)
pixel.clearCoat = material.clearCoat;
float clearCoatPerceptualRoughness = material.clearCoatRoughness;
clearCoatPerceptualRoughness =
clamp(clearCoatPerceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
#if defined(GEOMETRIC_SPECULAR_AA)
clearCoatPerceptualRoughness =
normalFiltering(clearCoatPerceptualRoughness, getWorldGeometricNormalVector());
#endif
pixel.clearCoatPerceptualRoughness = clearCoatPerceptualRoughness;
pixel.clearCoatRoughness = perceptualRoughnessToRoughness(clearCoatPerceptualRoughness);
#if defined(CLEAR_COAT_IOR_CHANGE)
pixel.f0 = mix(pixel.f0, f0ClearCoatToSurface(pixel.f0), pixel.clearCoat);
#endif
#endif
}
void getRoughnessPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
float perceptualRoughness = computeRoughnessFromGlossiness(material.glossiness);
#else
float perceptualRoughness = material.roughness;
#endif
pixel.perceptualRoughnessUnclamped = perceptualRoughness;
#if defined(GEOMETRIC_SPECULAR_AA)
perceptualRoughness = normalFiltering(perceptualRoughness, getWorldGeometricNormalVector());
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_ROUGHNESS)
float basePerceptualRoughness = max(perceptualRoughness, pixel.clearCoatPerceptualRoughness);
perceptualRoughness = mix(perceptualRoughness, basePerceptualRoughness, pixel.clearCoat);
#endif
pixel.perceptualRoughness = clamp(perceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
pixel.roughness = perceptualRoughnessToRoughness(pixel.perceptualRoughness);
}
void getSubsurfacePixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(SHADING_MODEL_SUBSURFACE)
pixel.subsurfacePower = material.subsurfacePower;
pixel.subsurfaceColor = material.subsurfaceColor;
pixel.thickness = saturate(material.thickness);
#endif
}
void getEnergyCompensationPixelParams(inout PixelParams pixel) {
pixel.dfg = prefilteredDFG(pixel.perceptualRoughness, shading_NoV);
#if !defined(SHADING_MODEL_CLOTH)
pixel.energyCompensation = 1.0 + pixel.f0 * (1.0 / pixel.dfg.y - 1.0);
#else
pixel.energyCompensation = vec3(1.0);
#endif
#if !defined(SHADING_MODEL_CLOTH)
#if defined(MATERIAL_HAS_SHEEN_COLOR)
pixel.sheenDFG = prefilteredDFG(pixel.sheenPerceptualRoughness, shading_NoV).z;
pixel.sheenScaling = 1.0 - max3(pixel.sheenColor) * pixel.sheenDFG;
#endif
#endif
}
void getPixelParams(const MaterialInputs material, out PixelParams pixel) {
getCommonPixelParams(material, pixel);
getSheenPixelParams(material, pixel);
getClearCoatPixelParams(material, pixel);
getRoughnessPixelParams(material, pixel);
getSubsurfacePixelParams(material, pixel);
getAnisotropyPixelParams(material, pixel);
getEnergyCompensationPixelParams(pixel);
}
vec4 evaluateLights(const MaterialInputs material) {
PixelParams pixel;
getPixelParams(material, pixel);
vec3 color = vec3(0.0);
evaluateIBL(material, pixel, color);
#if defined(VARIANT_HAS_DIRECTIONAL_LIGHTING)
evaluateDirectionalLight(material, pixel, color);
#endif
#if defined(VARIANT_HAS_DYNAMIC_LIGHTING)
evaluatePunctualLights(material, pixel, color);
#endif
#if defined(BLEND_MODE_FADE) && !defined(SHADING_MODEL_UNLIT)
color *= material.baseColor.a;
#endif
return vec4(color, computeDiffuseAlpha(material.baseColor.a));
}
void addEmissive(const MaterialInputs material, inout vec4 color) {
#if defined(MATERIAL_HAS_EMISSIVE)
highp vec4 emissive = material.emissive;
highp float attenuation = mix(1.0, getExposure(), emissive.w);
color.rgb += emissive.rgb * (attenuation * color.a);
#endif
}
vec4 evaluateMaterial(const MaterialInputs material) {
vec4 color = evaluateLights(material);
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

