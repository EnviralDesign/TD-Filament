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
#define POST_PROCESS_OPAQUE 1

LAYOUT_LOCATION(0) in highp vec4 variable_vertex;

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

#define FRAG_OUTPUT0 color

#define FRAG_OUTPUT_AT0 output_color

#define FRAG_OUTPUT_MATERIAL_TYPE0 float4

#define FRAG_OUTPUT_TYPE0 float4

#define FRAG_OUTPUT_SWIZZLE0 
layout(location=0) out float4 output_color;

struct PostProcessInputs {
#if defined(FRAG_OUTPUT0)
FRAG_OUTPUT_MATERIAL_TYPE0 FRAG_OUTPUT0;
#endif
#if defined(FRAG_OUTPUT1)
FRAG_OUTPUT_MATERIAL_TYPE1 FRAG_OUTPUT1;
#endif
#if defined(FRAG_OUTPUT2)
FRAG_OUTPUT_MATERIAL_TYPE2 FRAG_OUTPUT2;
#endif
#if defined(FRAG_OUTPUT3)
FRAG_OUTPUT_MATERIAL_TYPE3 FRAG_OUTPUT3;
#endif
#if defined(FRAG_OUTPUT4)
FRAG_OUTPUT_MATERIAL_TYPE4 FRAG_OUTPUT4;
#endif
#if defined(FRAG_OUTPUT5)
FRAG_OUTPUT_MATERIAL_TYPE5 FRAG_OUTPUT5;
#endif
#if defined(FRAG_OUTPUT6)
FRAG_OUTPUT_MATERIAL_TYPE6 FRAG_OUTPUT6;
#endif
#if defined(FRAG_OUTPUT7)
FRAG_OUTPUT_MATERIAL_TYPE7 FRAG_OUTPUT7;
#endif
#if defined(FRAG_OUTPUT_DEPTH)
float depth;
#endif
};
#line 46
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 45 "separableGaussianBlur1L.mat"
#endif


#define BLUR_TYPE    vec2
#define BLUR_SWIZZLE r
#define TEXTURE_LOD(s, p, m, l) textureLod(s, vec3(p, l), m)

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0 "separableGaussianBlur.fs"
#endif
// BLUR_TYPE and BLUR_SWIZZLE must be defined
// BLUR_TYPE        vec2, vec3, vec4
// BLUR_SWIZZLE     r, rg, rgb, rgba

float vmax(const float v) {
    return v;
}

void tap(inout highp BLUR_TYPE sum, const float weight, const highp vec2 position) {
    vec4 s = TEXTURE_LOD(materialParams_source, position, materialParams.level, materialParams.layer);
    sum.BLUR_SWIZZLE += s.BLUR_SWIZZLE * weight;
}

void tapReinhard(inout highp BLUR_TYPE sum, inout float totalWeight, const float weight, const highp vec2 position) {
    vec4 s = TEXTURE_LOD(materialParams_source, position, materialParams.level, materialParams.layer);
    float w = weight / (1.0 + vmax(s.BLUR_SWIZZLE));
    totalWeight += w;
    sum.BLUR_SWIZZLE += s.BLUR_SWIZZLE * w;
}

void postProcess(inout PostProcessInputs postProcess) {
    highp vec2 uv = variable_vertex.xy;

    // we handle the center pixel separately
    highp BLUR_TYPE sum = BLUR_TYPE(0.0);

    if (materialParams.reinhard != 0) {
        float totalWeight = 0.0;
        tapReinhard(sum, totalWeight, materialParams.kernel[0].x, uv);
        vec2 offset = materialParams.axis;
        for (int i = 1; i < materialParams.count; i++, offset += materialParams.axis * 2.0) {
            float k = materialParams.kernel[i].x;
            vec2 o = offset + materialParams.axis * materialParams.kernel[i].y;
            tapReinhard(sum, totalWeight, k, uv + o);
            tapReinhard(sum, totalWeight, k, uv - o);
        }
        sum *= 1.0 / totalWeight;
    } else {
        tap(sum, materialParams.kernel[0].x, uv);
        vec2 offset = materialParams.axis;
        for (int i = 1; i < materialParams.count; i++, offset += materialParams.axis * 2.0) {
            float k = materialParams.kernel[i].x;
            vec2 o = offset + materialParams.axis * materialParams.kernel[i].y;
            tap(sum, k, uv + o);
            tap(sum, k, uv - o);
        }
    }
    postProcess.color.BLUR_SWIZZLE = sum.BLUR_SWIZZLE;
}

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 52 "separableGaussianBlur1L.mat"
#endif

#line 499
void main() {
PostProcessInputs inputs;
postProcess(inputs);
#if defined(TARGET_MOBILE)
#if defined(FRAG_OUTPUT0)
inputs.FRAG_OUTPUT0 = clamp(inputs.FRAG_OUTPUT0, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT1)
inputs.FRAG_OUTPUT1 = clamp(inputs.FRAG_OUTPUT1, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT2)
inputs.FRAG_OUTPUT2 = clamp(inputs.FRAG_OUTPUT2, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT3)
inputs.FRAG_OUTPUT3 = clamp(inputs.FRAG_OUTPUT3, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT4)
inputs.FRAG_OUTPUT4 = clamp(inputs.FRAG_OUTPUT4, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT5)
inputs.FRAG_OUTPUT5 = clamp(inputs.FRAG_OUTPUT5, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT6)
inputs.FRAG_OUTPUT6 = clamp(inputs.FRAG_OUTPUT6, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#if defined(FRAG_OUTPUT7)
inputs.FRAG_OUTPUT7 = clamp(inputs.FRAG_OUTPUT7, -MEDIUMP_FLT_MAX, MEDIUMP_FLT_MAX);
#endif
#endif
#if defined(FRAG_OUTPUT0)
FRAG_OUTPUT_AT0 FRAG_OUTPUT_SWIZZLE0 = inputs.FRAG_OUTPUT0;
#endif
#if defined(FRAG_OUTPUT1)
FRAG_OUTPUT_AT1 FRAG_OUTPUT_SWIZZLE1 = inputs.FRAG_OUTPUT1;
#endif
#if defined(FRAG_OUTPUT2)
FRAG_OUTPUT_AT2 FRAG_OUTPUT_SWIZZLE2 = inputs.FRAG_OUTPUT2;
#endif
#if defined(FRAG_OUTPUT3)
FRAG_OUTPUT_AT3 FRAG_OUTPUT_SWIZZLE3 = inputs.FRAG_OUTPUT3;
#endif
#if defined(FRAG_OUTPUT4)
FRAG_OUTPUT_AT4 FRAG_OUTPUT_SWIZZLE4 = inputs.FRAG_OUTPUT4;
#endif
#if defined(FRAG_OUTPUT5)
FRAG_OUTPUT_AT5 FRAG_OUTPUT_SWIZZLE5 = inputs.FRAG_OUTPUT5;
#endif
#if defined(FRAG_OUTPUT6)
FRAG_OUTPUT_AT6 FRAG_OUTPUT_SWIZZLE6 = inputs.FRAG_OUTPUT6;
#endif
#if defined(FRAG_OUTPUT7)
FRAG_OUTPUT_AT7 FRAG_OUTPUT_SWIZZLE7 = inputs.FRAG_OUTPUT7;
#endif
#if defined(FRAG_OUTPUT_DEPTH)
gl_FragDepth = inputs.depth;
#endif
}

