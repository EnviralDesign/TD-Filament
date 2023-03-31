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
#define VARIANT_HAS_SKINNING_OR_MORPHING
#define VARIANT_HAS_VSM
#define SHADING_MODEL_UNLIT
#define SHADING_INTERPOLATION 
#define HAS_ATTRIBUTE_UV0
#define HAS_ATTRIBUTE_BONE_INDICES
#define HAS_ATTRIBUTE_BONE_WEIGHTS

#define LOCATION_POSITION 0
#define LOCATION_UV0 3
#define LOCATION_BONE_INDICES 5
#define LOCATION_BONE_WEIGHTS 6

layout(location = LOCATION_POSITION) in vec4 mesh_position;
#if defined(HAS_ATTRIBUTE_TANGENTS)
layout(location = LOCATION_TANGENTS) in vec4 mesh_tangents;
#endif
#if defined(HAS_ATTRIBUTE_COLOR)
layout(location = LOCATION_COLOR) in vec4 mesh_color;
#endif
#if defined(HAS_ATTRIBUTE_UV0)
layout(location = LOCATION_UV0) in vec2 mesh_uv0;
#endif
#if defined(HAS_ATTRIBUTE_UV1)
layout(location = LOCATION_UV1) in vec2 mesh_uv1;
#endif
#if defined(HAS_ATTRIBUTE_BONE_INDICES)
layout(location = LOCATION_BONE_INDICES) in uvec4 mesh_bone_indices;
#endif
#if defined(HAS_ATTRIBUTE_BONE_WEIGHTS)
layout(location = LOCATION_BONE_WEIGHTS) in vec4 mesh_bone_weights;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM0)
layout(location = LOCATION_CUSTOM0) in vec4 mesh_custom0;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM1)
layout(location = LOCATION_CUSTOM1) in vec4 mesh_custom1;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM2)
layout(location = LOCATION_CUSTOM2) in vec4 mesh_custom2;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM3)
layout(location = LOCATION_CUSTOM3) in vec4 mesh_custom3;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM4)
layout(location = LOCATION_CUSTOM4) in vec4 mesh_custom4;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM5)
layout(location = LOCATION_CUSTOM5) in vec4 mesh_custom5;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM6)
layout(location = LOCATION_CUSTOM6) in vec4 mesh_custom6;
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM7)
layout(location = LOCATION_CUSTOM7) in vec4 mesh_custom7;
#endif
#define VARYING out

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
#define VERTEX_DOMAIN_OBJECT

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

layout(binding = 2, std140) uniform BonesUniforms {
    BoneData bones[256];
} bonesUniforms;

layout(binding = 3, std140) uniform MorphingUniforms {
    lowp vec4 weights[256];
} morphingUniforms;
uniform highp sampler2DArray morphTargetBuffer_positions;
uniform highp isampler2DArray morphTargetBuffer_tangents;


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

mat4 getWorldFromModelMatrix() {
return object_uniforms.worldFromModelMatrix;
}
mat3 getWorldFromModelNormalMatrix() {
return object_uniforms.worldFromModelNormalMatrix;
}
float getObjectUserData() {
return object_uniforms.userData;
}
int getVertexIndex() {
#if defined(TARGET_METAL_ENVIRONMENT) || defined(TARGET_VULKAN_ENVIRONMENT)
return gl_VertexIndex;
#else
return gl_VertexID;
#endif
}
#if defined(VARIANT_HAS_SKINNING_OR_MORPHING)
vec3 mulBoneNormal(vec3 n, uint i) {
highp mat3 cof;
highp vec2 x0y0 = unpackHalf2x16(bonesUniforms.bones[i].cof[0]);
highp vec2 z0x1 = unpackHalf2x16(bonesUniforms.bones[i].cof[1]);
highp vec2 y1z1 = unpackHalf2x16(bonesUniforms.bones[i].cof[2]);
highp vec2 x2y2 = unpackHalf2x16(bonesUniforms.bones[i].cof[3]);
highp float a = bonesUniforms.bones[i].transform[0][0];
highp float b = bonesUniforms.bones[i].transform[0][1];
highp float d = bonesUniforms.bones[i].transform[1][0];
highp float e = bonesUniforms.bones[i].transform[1][1];
cof[0].xyz = vec3(x0y0, z0x1.x);
cof[1].xyz = vec3(z0x1.y, y1z1);
cof[2].xyz = vec3(x2y2, a * e - b * d);
return normalize(cof * n);
}
vec3 mulBoneVertex(vec3 v, uint i) {
highp mat4x3 m = transpose(bonesUniforms.bones[i].transform);
return v.x * m[0].xyz + (v.y * m[1].xyz + (v.z * m[2].xyz + m[3].xyz));
}
void skinNormal(inout vec3 n, const uvec4 ids, const vec4 weights) {
n =   mulBoneNormal(n, ids.x) * weights.x
+ mulBoneNormal(n, ids.y) * weights.y
+ mulBoneNormal(n, ids.z) * weights.z
+ mulBoneNormal(n, ids.w) * weights.w;
}
void skinPosition(inout vec3 p, const uvec4 ids, const vec4 weights) {
p =   mulBoneVertex(p, ids.x) * weights.x
+ mulBoneVertex(p, ids.y) * weights.y
+ mulBoneVertex(p, ids.z) * weights.z
+ mulBoneVertex(p, ids.w) * weights.w;
}
#define MAX_MORPH_TARGET_BUFFER_WIDTH 2048
void morphPosition(inout vec4 p) {
ivec3 texcoord = ivec3(getVertexIndex() % MAX_MORPH_TARGET_BUFFER_WIDTH, getVertexIndex() / MAX_MORPH_TARGET_BUFFER_WIDTH, 0);
uint c = object_uniforms.morphTargetCount;
for (uint i = 0u; i < c; ++i) {
float w = morphingUniforms.weights[i][0];
if (w != 0.0) {
texcoord.z = int(i);
p += w * texelFetch(morphTargetBuffer_positions, texcoord, 0);
}
}
}
void morphNormal(inout vec3 n) {
vec3 baseNormal = n;
ivec3 texcoord = ivec3(getVertexIndex() % MAX_MORPH_TARGET_BUFFER_WIDTH, getVertexIndex() / MAX_MORPH_TARGET_BUFFER_WIDTH, 0);
uint c = object_uniforms.morphTargetCount;
for (uint i = 0u; i < c; ++i) {
float w = morphingUniforms.weights[i][0];
if (w != 0.0) {
texcoord.z = int(i);
ivec4 tangent = texelFetch(morphTargetBuffer_tangents, texcoord, 0);
vec3 normal;
toTangentFrame(float4(tangent) * (1.0 / 32767.0), normal);
n += w * (normal - baseNormal);
}
}
}
#endif
vec4 getPosition() {
vec4 pos = mesh_position;
#if defined(VARIANT_HAS_SKINNING_OR_MORPHING)
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_MORPHING_ENABLED_BIT) != 0u) {
#if defined(LEGACY_MORPHING)
pos += morphingUniforms.weights[0] * mesh_custom0;
pos += morphingUniforms.weights[1] * mesh_custom1;
pos += morphingUniforms.weights[2] * mesh_custom2;
pos += morphingUniforms.weights[3] * mesh_custom3;
#else
morphPosition(pos);
#endif
}
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_SKINNING_ENABLED_BIT) != 0u) {
skinPosition(pos.xyz, mesh_bone_indices, mesh_bone_weights);
}
#endif
return pos;
}
#if defined(HAS_ATTRIBUTE_CUSTOM0)
vec4 getCustom0() { return mesh_custom0; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM1)
vec4 getCustom1() { return mesh_custom1; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM2)
vec4 getCustom2() { return mesh_custom2; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM3)
vec4 getCustom3() { return mesh_custom3; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM4)
vec4 getCustom4() { return mesh_custom4; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM5)
vec4 getCustom5() { return mesh_custom5; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM6)
vec4 getCustom6() { return mesh_custom6; }
#endif
#if defined(HAS_ATTRIBUTE_CUSTOM7)
vec4 getCustom7() { return mesh_custom7; }
#endif
vec4 computeWorldPosition() {
#if defined(VERTEX_DOMAIN_OBJECT)
mat4 transform = getWorldFromModelMatrix();
vec3 position = getPosition().xyz;
return mulMat4x4Float3(transform, position);
#elif defined(VERTEX_DOMAIN_WORLD)
return vec4(getPosition().xyz, 1.0);
#elif defined(VERTEX_DOMAIN_VIEW)
mat4 transform = getWorldFromViewMatrix();
vec3 position = getPosition().xyz;
return mulMat4x4Float3(transform, position);
#elif defined(VERTEX_DOMAIN_DEVICE)
mat4 transform = getWorldFromClipMatrix();
vec4 p = getPosition();
p.z = p.z * -0.5 + 0.5;
vec4 position = transform * p;
if (abs(position.w) < MEDIUMP_FLT_MIN) {
position.w = position.w < 0.0 ? -MEDIUMP_FLT_MIN : MEDIUMP_FLT_MIN;
}
return position * (1.0 / position.w);
#else
#error Unknown Vertex Domain
#endif
}
struct MaterialVertexInputs {
#ifdef HAS_ATTRIBUTE_COLOR
vec4 color;
#endif
#ifdef HAS_ATTRIBUTE_UV0
vec2 uv0;
#endif
#ifdef HAS_ATTRIBUTE_UV1
vec2 uv1;
#endif
#ifdef VARIABLE_CUSTOM0
vec4 VARIABLE_CUSTOM0;
#endif
#ifdef VARIABLE_CUSTOM1
vec4 VARIABLE_CUSTOM1;
#endif
#ifdef VARIABLE_CUSTOM2
vec4 VARIABLE_CUSTOM2;
#endif
#ifdef VARIABLE_CUSTOM3
vec4 VARIABLE_CUSTOM3;
#endif
#ifdef HAS_ATTRIBUTE_TANGENTS
vec3 worldNormal;
#endif
vec4 worldPosition;
#ifdef VERTEX_DOMAIN_DEVICE
#ifdef MATERIAL_HAS_CLIP_SPACE_TRANSFORM
mat4 clipSpaceTransform;
#endif
#endif
};
vec4 getWorldPosition(const MaterialVertexInputs material) {
return material.worldPosition;
}
#ifdef VERTEX_DOMAIN_DEVICE
#ifdef MATERIAL_HAS_CLIP_SPACE_TRANSFORM
mat4 getMaterialClipSpaceTransform(const MaterialVertexInputs material) {
return material.clipSpaceTransform;
}
#endif
#endif
void initMaterialVertex(out MaterialVertexInputs material) {
#ifdef HAS_ATTRIBUTE_COLOR
material.color = mesh_color;
#endif
#ifdef HAS_ATTRIBUTE_UV0
#ifdef FLIP_UV_ATTRIBUTE
material.uv0 = vec2(mesh_uv0.x, 1.0 - mesh_uv0.y);
#else
material.uv0 = mesh_uv0;
#endif
#endif
#ifdef HAS_ATTRIBUTE_UV1
#ifdef FLIP_UV_ATTRIBUTE
material.uv1 = vec2(mesh_uv1.x, 1.0 - mesh_uv1.y);
#else
material.uv1 = mesh_uv1;
#endif
#endif
#ifdef VARIABLE_CUSTOM0
material.VARIABLE_CUSTOM0 = vec4(0.0);
#endif
#ifdef VARIABLE_CUSTOM1
material.VARIABLE_CUSTOM1 = vec4(0.0);
#endif
#ifdef VARIABLE_CUSTOM2
material.VARIABLE_CUSTOM2 = vec4(0.0);
#endif
#ifdef VARIABLE_CUSTOM3
material.VARIABLE_CUSTOM3 = vec4(0.0);
#endif
material.worldPosition = computeWorldPosition();
#ifdef VERTEX_DOMAIN_DEVICE
#ifdef MATERIAL_HAS_CLIP_SPACE_TRANSFORM
material.clipSpaceTransform = mat4(1.0);
#endif
#endif
}
#line 144
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 143 "unlit.mat"
#endif

  void materialVertex(inout MaterialVertexInputs material) {
    }
#line 683

void main() {
#if defined(TARGET_METAL_ENVIRONMENT) || defined(TARGET_VULKAN_ENVIRONMENT)
instance_index = gl_InstanceIndex;
#else
instance_index = gl_InstanceID;
#endif
initObjectUniforms(object_uniforms);
#if defined(USE_OPTIMIZED_DEPTH_VERTEX_SHADER)
#if !defined(VERTEX_DOMAIN_DEVICE) || defined(VARIANT_HAS_VSM)
MaterialVertexInputs material;
initMaterialVertex(material);
materialVertex(material);
#endif
#else
MaterialVertexInputs material;
initMaterialVertex(material);
#if defined(HAS_ATTRIBUTE_TANGENTS)
#if defined(MATERIAL_NEEDS_TBN)
toTangentFrame(mesh_tangents, material.worldNormal, vertex_worldTangent.xyz);
#if defined(VARIANT_HAS_SKINNING_OR_MORPHING)
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_MORPHING_ENABLED_BIT) != 0u) {
#if defined(LEGACY_MORPHING)
vec3 normal0, normal1, normal2, normal3;
toTangentFrame(mesh_custom4, normal0);
toTangentFrame(mesh_custom5, normal1);
toTangentFrame(mesh_custom6, normal2);
toTangentFrame(mesh_custom7, normal3);
vec3 baseNormal = material.worldNormal;
material.worldNormal += morphingUniforms.weights[0].xyz * (normal0 - baseNormal);
material.worldNormal += morphingUniforms.weights[1].xyz * (normal1 - baseNormal);
material.worldNormal += morphingUniforms.weights[2].xyz * (normal2 - baseNormal);
material.worldNormal += morphingUniforms.weights[3].xyz * (normal3 - baseNormal);
#else
morphNormal(material.worldNormal);
material.worldNormal = normalize(material.worldNormal);
#endif
}
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_SKINNING_ENABLED_BIT) != 0u) {
skinNormal(material.worldNormal, mesh_bone_indices, mesh_bone_weights);
skinNormal(vertex_worldTangent.xyz, mesh_bone_indices, mesh_bone_weights);
}
#endif
vertex_worldTangent.xyz = getWorldFromModelNormalMatrix() * vertex_worldTangent.xyz;
vertex_worldTangent.w = mesh_tangents.w;
material.worldNormal = getWorldFromModelNormalMatrix() * material.worldNormal;
#else
toTangentFrame(mesh_tangents, material.worldNormal);
#if defined(VARIANT_HAS_SKINNING_OR_MORPHING)
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_MORPHING_ENABLED_BIT) != 0u) {
#if defined(LEGACY_MORPHING)
vec3 normal0, normal1, normal2, normal3;
toTangentFrame(mesh_custom4, normal0);
toTangentFrame(mesh_custom5, normal1);
toTangentFrame(mesh_custom6, normal2);
toTangentFrame(mesh_custom7, normal3);
vec3 baseNormal = material.worldNormal;
material.worldNormal += morphingUniforms.weights[0].xyz * (normal0 - baseNormal);
material.worldNormal += morphingUniforms.weights[1].xyz * (normal1 - baseNormal);
material.worldNormal += morphingUniforms.weights[2].xyz * (normal2 - baseNormal);
material.worldNormal += morphingUniforms.weights[3].xyz * (normal3 - baseNormal);
#else
morphNormal(material.worldNormal);
material.worldNormal = normalize(material.worldNormal);
#endif
}
if ((object_uniforms.flagsChannels & FILAMENT_OBJECT_SKINNING_ENABLED_BIT) != 0u) {
skinNormal(material.worldNormal, mesh_bone_indices, mesh_bone_weights);
}
#endif
material.worldNormal = getWorldFromModelNormalMatrix() * material.worldNormal;
#endif
#endif
materialVertex(material);
#if defined(HAS_ATTRIBUTE_COLOR)
vertex_color = material.color;
#endif
#if defined(HAS_ATTRIBUTE_UV0)
vertex_uv01.xy = material.uv0;
#endif
#if defined(HAS_ATTRIBUTE_UV1)
vertex_uv01.zw = material.uv1;
#endif
#if defined(VARIABLE_CUSTOM0)
VARIABLE_CUSTOM_AT0 = material.VARIABLE_CUSTOM0;
#endif
#if defined(VARIABLE_CUSTOM1)
VARIABLE_CUSTOM_AT1 = material.VARIABLE_CUSTOM1;
#endif
#if defined(VARIABLE_CUSTOM2)
VARIABLE_CUSTOM_AT2 = material.VARIABLE_CUSTOM2;
#endif
#if defined(VARIABLE_CUSTOM3)
VARIABLE_CUSTOM_AT3 = material.VARIABLE_CUSTOM3;
#endif
vertex_worldPosition.xyz = material.worldPosition.xyz;
#ifdef HAS_ATTRIBUTE_TANGENTS
vertex_worldNormal = material.worldNormal;
#endif
#if defined(VARIANT_HAS_SHADOWING) && defined(VARIANT_HAS_DIRECTIONAL_LIGHTING)
vertex_lightSpacePosition = computeLightSpacePosition(
vertex_worldPosition.xyz, vertex_worldNormal,
frameUniforms.lightDirection,
shadowUniforms.shadows[0].normalBias,
shadowUniforms.shadows[0].lightFromWorldMatrix);
#endif
#endif
#if defined(VERTEX_DOMAIN_DEVICE)
gl_Position = getPosition();
#if !defined(USE_OPTIMIZED_DEPTH_VERTEX_SHADER)
#if defined(MATERIAL_HAS_CLIP_SPACE_TRANSFORM)
gl_Position = getMaterialClipSpaceTransform(material) * gl_Position;
#endif
#endif
#if defined(MATERIAL_HAS_VERTEX_DOMAIN_DEVICE_JITTERED)
gl_Position.xy = gl_Position.xy * frameUniforms.clipTransform.xy + (gl_Position.w * frameUniforms.clipTransform.zw);
#endif
#else
gl_Position = getClipFromWorldMatrix() * getWorldPosition(material);
#endif
#if defined(VERTEX_DOMAIN_DEVICE)
gl_Position.z = gl_Position.z * -0.5 + 0.5;
#endif
#if defined(VARIANT_HAS_VSM)
highp float z = (getViewFromWorldMatrix() * getWorldPosition(material)).z;
highp float depth = -z * frameUniforms.oneOverFarMinusNear - frameUniforms.nearOverFarMinusNear;
depth = depth * 2.0 - 1.0;
vertex_worldPosition.w = depth;
#endif
vertex_position = gl_Position;
#if defined(TARGET_VULKAN_ENVIRONMENT)
gl_Position.y = -gl_Position.y;
#endif
#if !defined(TARGET_VULKAN_ENVIRONMENT) && !defined(TARGET_METAL_ENVIRONMENT)
gl_Position.z = dot(gl_Position.zw, frameUniforms.clipControl);
#endif
}

