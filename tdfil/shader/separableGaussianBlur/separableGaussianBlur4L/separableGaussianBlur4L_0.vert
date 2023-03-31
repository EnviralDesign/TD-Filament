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
#define LOCATION_POSITION 0

#define VARIABLE_CUSTOM0 vertex

#define VARIABLE_CUSTOM_AT0 variable_vertex
LAYOUT_LOCATION(0) out vec4 variable_vertex;
LAYOUT_LOCATION(LOCATION_POSITION) in vec4 position;
struct PostProcessVertexInputs {
vec2 normalizedUV;
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
};
void initPostProcessMaterialVertex(out PostProcessVertexInputs inputs) {
#ifdef VARIABLE_CUSTOM0
inputs.VARIABLE_CUSTOM0 = vec4(0.0);
#endif
#ifdef VARIABLE_CUSTOM1
inputs.VARIABLE_CUSTOM1 = vec4(0.0);
#endif
#ifdef VARIABLE_CUSTOM2
inputs.VARIABLE_CUSTOM2 = vec4(0.0);
#endif
#ifdef VARIABLE_CUSTOM3
inputs.VARIABLE_CUSTOM3 = vec4(0.0);
#endif
}
#define POST_PROCESS_OPAQUE 1

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

layout(std140) uniform MaterialParams {
    vec2 axis;
    float level;
    float layer;
    int count;
    int reinhard;
    vec2 kernel[32];
} materialParams;
uniform mediump sampler2DArray materialParams_source;


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

vec4 getPosition() {
vec4 pos = position;
#if defined(TARGET_VULKAN_ENVIRONMENT)
pos.y = -pos.y;
#endif
return pos;
}
#line 42
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 41 "separableGaussianBlur4L.mat"
#endif

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0 "separableGaussianBlur.vs"
#endif
void postProcessVertex(inout PostProcessVertexInputs postProcess) {
    postProcess.vertex.xy = uvToRenderTargetUV(postProcess.normalizedUV);
}

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 43 "separableGaussianBlur4L.mat"
#endif
#line 373
void main() {
PostProcessVertexInputs inputs;
initPostProcessMaterialVertex(inputs);
inputs.normalizedUV = position.xy * 0.5 + 0.5;
gl_Position = getPosition();
gl_Position.z = gl_Position.z * -0.5 + 0.5;
#if !defined(TARGET_VULKAN_ENVIRONMENT) && !defined(TARGET_METAL_ENVIRONMENT)
gl_Position.z = dot(gl_Position.zw, frameUniforms.clipControl);
#endif
postProcessVertex(inputs);
#if defined(VARIABLE_CUSTOM0)
VARIABLE_CUSTOM_AT0 = inputs.VARIABLE_CUSTOM0;
#endif
#if defined(VARIABLE_CUSTOM1)
VARIABLE_CUSTOM_AT1 = inputs.VARIABLE_CUSTOM1;
#endif
#if defined(VARIABLE_CUSTOM2)
VARIABLE_CUSTOM_AT2 = inputs.VARIABLE_CUSTOM2;
#endif
#if defined(VARIABLE_CUSTOM3)
VARIABLE_CUSTOM_AT3 = inputs.VARIABLE_CUSTOM3;
#endif
}

