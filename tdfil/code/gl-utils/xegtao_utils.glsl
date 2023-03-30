// #define XE_GTAO_USE_HALF_FLOAT_PRECISION

#define XE_GTAO_PI 3.1415926535897932384626433832795
#define XE_GTAO_PI_HALF 1.5707963267948966192313216916398
#define XE_GTAO_DEPTH_MIP_LEVELS 5
#define XE_GTAO_OCCLUSION_TERM_SCALE 1.5

float saturate(float value){return clamp(value,0.0,1.0);}
vec4 saturate(vec4 value){return clamp(value,vec4(0.0),vec4(1.0));}


// http://h14s.p5r.org/2012/09/0x5f3759df.html, [Drobot2014a] Low Level Optimizations for GCN, 
// https://blog.selfshadow.com/publications/s2016-shading-course/activision/s2016_pbs_activision_occlusion.pdf slide 63
float XeGTAO_FastSqrt( float x )
{
    return intBitsToFloat( 0x1fbd1df5 + ( floatBitsToInt( x ) >> 1 ) );
}


// input [-1, 1] and output [0, PI], from https://seblagarde.wordpress.com/2014/12/01/inverse-trigonometric-functions-gpu-optimization-for-amd-gcn-architecture/
float XeGTAO_FastACos( float inX )
{ 
    const float PI = 3.141593;
    const float HALF_PI = 1.570796;
    float x = abs(inX); 
    float res = -0.156583 * x + HALF_PI; 
    res *= XeGTAO_FastSqrt(1.0 - x); 
    return (inX >= 0) ? res : PI - res; 
}


// packing/unpacking for edges; 2 bits per edge mean 4 gradient values (0, 0.33, 0.66, 1) for smoother transitions!
float XeGTAO_PackEdges( vec4 edgesLRTB )
{
    // integer version:
    // edgesLRTB = saturate(edgesLRTB) * 2.9.xxxx + 0.5.xxxx;
    // return (((uint)edgesLRTB.x) << 6) + (((uint)edgesLRTB.y) << 4) + (((uint)edgesLRTB.z) << 2) + (((uint)edgesLRTB.w));
    // 
    // optimized, should be same as above
    edgesLRTB = round( saturate( edgesLRTB ) * 2.9 );
    return dot( edgesLRTB, vec4( 64.0 / 255.0, 16.0 / 255.0, 4.0 / 255.0, 1.0 / 255.0 ) ) ;
}


vec4 XeGTAO_UnpackEdges( float _packedVal )
{
    uint packedVal = uint(_packedVal * 255.5);
    vec4 edgesLRTB;
    edgesLRTB.x = float((packedVal >> 6) & 0x03) / 3.0;          // there's really no need for mask (as it's an 8 bit input) but I'll leave it in so it doesn't cause any trouble in the future
    edgesLRTB.y = float((packedVal >> 4) & 0x03) / 3.0;
    edgesLRTB.z = float((packedVal >> 2) & 0x03) / 3.0;
    edgesLRTB.w = float((packedVal >> 0) & 0x03) / 3.0;

    return saturate( edgesLRTB );
}


// this function in the original material expected a depth value that was in NDC space aka -1:1 range.
// however since TD's default depth is 0-1 (screen space depth) I opted for doing the transformation to NDC depth here
// to keep the transformations neccesary in other tops to a minimum and leave it all up to this shader system instead. 
float XeGTAO_ScreenSpaceToViewSpaceDepth( const float screenSpaceDepth, const float depthLinearizeMul, const float depthLinearizeAdd )
{
    float NdcSpaceDepth = screenSpaceDepth * 2 - 1;
    // Optimised version of "-cameraClipNear / (cameraClipFar - projDepth * (cameraClipFar - cameraClipNear)) * cameraClipFar"
    // return 0.01 / (100000.0 - NdcSpaceDepth * (100000.0 - 0.01)) * 100000.0;
    return depthLinearizeMul / (depthLinearizeAdd - NdcSpaceDepth);
}


// This is also a good place to do non-linear depth conversion for cases where one wants the 'radius' (effectively the threshold between near-field and far-field GI), 
// is required to be non-linear (i.e. very large outdoors environments).
float XeGTAO_ClampDepth( float depth )
{
#ifdef XE_GTAO_USE_HALF_FLOAT_PRECISION
    return clamp( depth, 0.0, 65504.0 );
#else
    return clamp( depth, 0.0, 3.402823466e+38 );
#endif
}


// Inputs are screen XY and viewspace depth, output is viewspace position
// screen XY is normalized 0-1 coordinates, so in TD just use vUV.st.
// to get viewspace depth in TD, use a depth TOP, and set pixel format to 16 bit (or 32) and then Depth Space to "Camera Space"
// NDCToViewMul and NDCToViewAdd are values that should be calculated in chop/cpu land and passed in as uniforms. (see uniformutils.py)
vec3 XeGTAO_ComputeViewspacePosition( const vec2 screenPos, const float viewspaceDepth, const vec2 NDCToViewMul, const vec2 NDCToViewAdd )
{
    

    vec3 ret;
    ret.xy = (NDCToViewMul * screenPos.xy + NDCToViewAdd) * viewspaceDepth;
    ret.z = viewspaceDepth;

    return ret;
}


vec4 XeGTAO_CalculateEdges( const float centerZ, const float leftZ, const float rightZ, const float topZ, const float bottomZ )
{
    vec4 edgesLRTB = vec4( leftZ, rightZ, topZ, bottomZ ) - centerZ;

    float slopeLR = (edgesLRTB.y - edgesLRTB.x) * 0.5;
    float slopeTB = (edgesLRTB.w - edgesLRTB.z) * 0.5;
    vec4 edgesLRTBSlopeAdjusted = edgesLRTB + vec4( slopeLR, -slopeLR, slopeTB, -slopeTB );
    edgesLRTB = min( abs( edgesLRTB ), abs( edgesLRTBSlopeAdjusted ) );
    return vec4(saturate( ( 1.25 - edgesLRTB / (centerZ * 0.011) ) ));
}


void XeGTAO_DecodeGatherPartial( const uvec4 packedValue, out float outDecoded[4] )
{
    for( int i = 0; i < 4; i++ )
        outDecoded[i] = float(packedValue[i]) / float(255.0);
}


vec3 XeGTAO_CalculateNormal( const vec4 edgesLRTB, vec3 pixCenterPos, vec3 pixLPos, vec3 pixRPos, vec3 pixTPos, vec3 pixBPos )
{
    // Get this pixel's viewspace normal
    vec4 acceptedNormals  = saturate( vec4( edgesLRTB.x*edgesLRTB.z, edgesLRTB.z*edgesLRTB.y, edgesLRTB.y*edgesLRTB.w, edgesLRTB.w*edgesLRTB.x ) + 0.01 );

    pixLPos = normalize(pixLPos - pixCenterPos);
    pixRPos = normalize(pixRPos - pixCenterPos);
    pixTPos = normalize(pixTPos - pixCenterPos);
    pixBPos = normalize(pixBPos - pixCenterPos);

    vec3 pixelNormal =  acceptedNormals.x * cross( pixLPos, pixTPos ) +
                        + acceptedNormals.y * cross( pixTPos, pixRPos ) +
                        + acceptedNormals.z * cross( pixRPos, pixBPos ) +
                        + acceptedNormals.w * cross( pixBPos, pixLPos );
    pixelNormal = normalize( pixelNormal );

    return pixelNormal;
}


// Generic viewspace normal generate pass
// NOTE THis is currently broken...
vec3 XeGTAO_ComputeViewspaceNormal( const uvec2 pixCoord, sampler2D ScreenSpaceDepthSampler, const vec2 ViewportPixelSize, const float depthLinearizeMul, const float depthLinearizeAdd, const vec2 NDCToViewMul, const vec2 NDCToViewAdd )
{
    // can we just pass in vUV.st here?
    vec2 normalizedScreenPos = (pixCoord + vec2(0.5).xx) * ViewportPixelSize;
    /////// TODO; this might need some love.. do we need to flip vertical coord due to DirectX?

    // ORIGINAL CODE - confusing, this link sheds some light on what's actually happening:
    // http://wojtsterna.blogspot.com/2018/02/directx-11-hlsl-gatherred.html
    // float4 valuesUL   = sourceNDCDepth.GatherRed( depthSampler, float2( pixCoord * consts.ViewportPixelSize )               );
    // float4 valuesBR   = sourceNDCDepth.GatherRed( depthSampler, float2( pixCoord * consts.ViewportPixelSize ), int2( 1, 1 ) );

    // NOTE!! we are sampling SCREEN SPACE DEPTH aka 0-1 aka exactly how it comes in from TouchDesigner's depth TOP with default settings.
    // this is not how the original resources was written. the original samples NdcSpaceDepthSampler instead, which is very similar, but is depth in -1:1 space.
    vec4 valuesUL   =  textureGather( ScreenSpaceDepthSampler, vec2(  pixCoord                  * ViewportPixelSize ) , 0);
    vec4 valuesBR   =  textureGather( ScreenSpaceDepthSampler, vec2( (pixCoord + vec2( 1, 1 ))  * ViewportPixelSize ) , 0);

    // viewspace Z at the center
    float viewspaceZ  = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesUL.y, depthLinearizeMul, depthLinearizeAdd ); //sourceViewspaceDepth.SampleLevel( ScreenSpaceDepthSampler, normalizedScreenPos, 0 ).x; 

    // viewspace Zs left top right bottom
    float pixLZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesUL.x, depthLinearizeMul, depthLinearizeAdd );
    float pixTZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesUL.z, depthLinearizeMul, depthLinearizeAdd );
    float pixRZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesBR.z, depthLinearizeMul, depthLinearizeAdd );
    float pixBZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesBR.x, depthLinearizeMul, depthLinearizeAdd );

    vec4 edgesLRTB  = XeGTAO_CalculateEdges( viewspaceZ, pixLZ, pixRZ, pixTZ, pixBZ );

    vec3 CENTER   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ, NDCToViewMul, NDCToViewAdd );
    vec3 LEFT     = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + vec2(-1,  0) * ViewportPixelSize, pixLZ, NDCToViewMul, NDCToViewAdd );
    vec3 RIGHT    = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + vec2( 1,  0) * ViewportPixelSize, pixRZ, NDCToViewMul, NDCToViewAdd );
    vec3 TOP      = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + vec2( 0, -1) * ViewportPixelSize, pixTZ, NDCToViewMul, NDCToViewAdd );
    vec3 BOTTOM   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos + vec2( 0,  1) * ViewportPixelSize, pixBZ, NDCToViewMul, NDCToViewAdd );
    
    vec3 CamSpaceNormals = XeGTAO_CalculateNormal( edgesLRTB, CENTER, LEFT, RIGHT, TOP, BOTTOM );


    // shouldn't have to flip this here.. but it works.. TODO: figure out where the above port is going wrong, that we need to flip here.
    return -CamSpaceNormals;
}


// "Efficiently building a matrix to rotate one vector to another"
// http://cs.brown.edu/research/pubs/pdfs/1999/Moller-1999-EBA.pdf / https://dl.acm.org/doi/10.1080/10867651.1999.10487509
// (using https://github.com/assimp/assimp/blob/master/include/assimp/matrix3x3.inl#L275 as a code reference as it seems to be best)
mat3 XeGTAO_RotFromToMatrix( const vec3 from, const vec3 to )
{
    const float e       = dot(from, to);
    const float f       = abs(e); //(e < 0)? -e:e;

    // WARNING: This has not been tested/worked through, especially not for 16bit floats; seems to work in our special use case (from is always {0, 0, -1}) but wouldn't use it in general
    if( f > float( 1.0 - 0.0003 ) )
        return mat3( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    const vec3 v      = cross( from, to );
    /* ... use this hand optimized version (9 mults less) */
    const float h       = (1.0)/(1.0 + e);      /* optimization by Gottfried Chen */
    const float hvx     = h * v.x;
    const float hvz     = h * v.z;
    const float hvxy    = hvx * v.y;
    const float hvxz    = hvx * v.z;
    const float hvyz    = hvz * v.y;

    mat3 mtx;
    mtx[0][0] = e + hvx * v.x;
    mtx[0][1] = hvxy - v.z;
    mtx[0][2] = hvxz + v.y;

    mtx[1][0] = hvxy + v.z;
    mtx[1][1] = e + h * v.y * v.y;
    mtx[1][2] = hvyz - v.x;

    mtx[2][0] = hvxz - v.y;
    mtx[2][1] = hvyz + v.x;
    mtx[2][2] = e + hvz * v.z;

    return mtx;
}


void XeGTAO_OutputWorkingTerm( const uvec2 pixCoord, float visibility, vec3 bentNormal, inout float outWorkingAOTerm )
{
    visibility = saturate( visibility / float(XE_GTAO_OCCLUSION_TERM_SCALE) );
    outWorkingAOTerm = uint(visibility * 255.0 + 0.5);
}


// Engine-specific screen & temporal noise loader
// original reference shader : https://github.com/GameTechDev/XeGTAO/blob/master/Source/Rendering/Shaders/vaGTAO.hlsl
// line 73 - 91
vec2 SpatioTemporalNoise( uvec2 pixCoord, sampler2D HilbertLutSampler, uint temporalIndex )    // without TAA, temporalIndex is always 0
{
    uint index = uint(texelFetch( HilbertLutSampler , ivec2(pixCoord % 64) , 0 ).x);
    index += 288*(temporalIndex%64); // why 288? tried out a few and that's the best so far (with XE_HILBERT_LEVEL 6U) - but there's probably better :)
    // R2 sequence - see http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    return vec2( fract( 0.5 + index * vec2(0.75487766624669276005, 0.5698402909980532659114) ) );
}


void XeGTAO_OutputWorkingTerm(float visibility, vec3 bentNormal, inout uint outWorkingAOTerm )
{
#ifdef XE_DENOISE_ACTIVE
    visibility = saturate( visibility / float(XE_GTAO_OCCLUSION_TERM_SCALE) );
#else 
    visibility = visibility;
#endif

    outWorkingAOTerm = uint(visibility * 255.0 + 0.5);
}


void XeGTAO_AddSample( float ssaoValue, float edgeValue, inout float sum, inout float sumWeight )
{
    float weight = edgeValue;    

    sum += (weight * ssaoValue);
    sumWeight += weight;
}

// with vulcan or in TD, we get the error that says ERROR: /Engine/Viewport/XeGTAO/xegtao_utils:551: 'XeGTAO_Output' : no matching overloaded function found 
// since this is only ever called in the Denoise pass, we can just put the two lines of code in the denoise pass directly.
// void XeGTAO_Output(inout uint final_ao_value, float outputValue, const bool finalApply )
// {
//     outputValue *=  (finalApply)?(float(XE_GTAO_OCCLUSION_TERM_SCALE)):(1.0);
//     final_ao_value = uint(outputValue * 255.0 + 0.5);
// }


void XeGTAO_MainPass( const uvec2 pixCoord, const vec2 normUV, float sliceCount, float stepsPerSlice, const vec2 localNoise, vec3 viewspaceNormal,
    sampler2D ScreenSpaceDepthSampler, const vec2 ViewportPixelSize, const float depthLinearizeMul, 
    const float depthLinearizeAdd, const vec2 NDCToViewMul, const vec2 NDCToViewAdd, const vec2 NDCToViewMul_x_PixelSize, const float EffectRadius,
    const float RadiusMultiplier, const float SampleDistributionPower, const float ThinOccluderCompensation, const float EffectFalloffRange,
    const float DepthMIPSamplingOffset, const float FinalValuePower, inout uint outWorkingAOTerm, inout float outWorkingEdges)
{                                                                       
    // vec2 normalizedScreenPos = (pixCoord + vec2(0.5,0.5).xx) * ViewportPixelSize; // original
    vec2 normalizedScreenPos = normUV; // mine

    // GL screen space
    // TODO: can we optimize this by just using normalizedScreenPos and add the ViewportPixelSize for TR instead?
    vec4 valuesBL   =  textureGather( ScreenSpaceDepthSampler, vec2(  pixCoord                  * ViewportPixelSize ) , 0);
    vec4 valuesTR   =  textureGather( ScreenSpaceDepthSampler, vec2( (pixCoord + vec2( 1, 1 ))  * ViewportPixelSize ) , 0);
    
    //GL viewspace Z at the center
    float viewspaceZ  = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesBL.y, depthLinearizeMul, depthLinearizeAdd );

    //GL
    float pixLZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesBL.x, depthLinearizeMul, depthLinearizeAdd );
    float pixTZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesTR.x, depthLinearizeMul, depthLinearizeAdd );
    float pixRZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesTR.z, depthLinearizeMul, depthLinearizeAdd );
    float pixBZ = XeGTAO_ScreenSpaceToViewSpaceDepth( valuesBL.z, depthLinearizeMul, depthLinearizeAdd );

    vec4 edgesLRTB  = XeGTAO_CalculateEdges( viewspaceZ, pixLZ, pixRZ, pixTZ, pixBZ );
    outWorkingEdges = XeGTAO_PackEdges(edgesLRTB);

    vec3 pixCenterPos   = XeGTAO_ComputeViewspacePosition( normalizedScreenPos, viewspaceZ, NDCToViewMul, NDCToViewAdd );

    viewspaceZ *= 0.99920;     // this is good for FP16 depth buffer
    
    vec3 viewVec      = normalize(-pixCenterPos);
    const float effectRadius              = EffectRadius * RadiusMultiplier;
    const float sampleDistributionPower   = SampleDistributionPower;
    const float thinOccluderCompensation  = ThinOccluderCompensation;
    const float falloffRange              = EffectFalloffRange * effectRadius;
    const float falloffFrom       = effectRadius * (1-EffectFalloffRange);

    // fadeout precompute optimisation
    const float falloffMul        = -1.0 / falloffRange;
    const float falloffAdd        = falloffFrom / falloffRange + 1.0;

    float visibility = 0;
    vec3 bentNormal = viewspaceNormal;

    // see "Algorithm 1" in https://www.activision.com/cdn/research/Practical_Real_Time_Strategies_for_Accurate_Indirect_Occlusion_NEW%20VERSION_COLOR.pdf
    {
        const float noiseSlice  = localNoise.x;
        const float noiseSample = localNoise.y;

        // quality settings / tweaks / hacks
        const float pixelTooCloseThreshold  = 1.3;      // if the offset is under approx pixel size (pixelTooCloseThreshold), push it out to the minimum distance

        // approx viewspace pixel size at pixCoord; approximation of NDCToViewspace( normalizedScreenPos.xy + consts.ViewportPixelSize.xy, pixCenterPos.z ).xy - pixCenterPos.xy;
        vec2 pixelDirRBViewspaceSizeAtCenterZ = vec2(viewspaceZ) * NDCToViewMul_x_PixelSize;

        float screenspaceRadius   = effectRadius / pixelDirRBViewspaceSizeAtCenterZ.x;

        // fade out for small screen radii 
        visibility += saturate((10 - screenspaceRadius)/100)*0.5;


        // this is the min distance to start sampling from to avoid sampling from the center pixel (no useful data obtained from sampling center pixel)
        float minS = pixelTooCloseThreshold / screenspaceRadius;
        
        for( float slice = 0; slice < sliceCount; slice++ )
        {
            float sliceK = (slice+noiseSlice) / sliceCount;
            // lines 5, 6 from the paper
            float phi = sliceK * XE_GTAO_PI;
            float cosPhi = cos(phi);
            float sinPhi = sin(phi);
            vec2 omega = vec2(cosPhi, -sinPhi);       //lpfloat2 on omega causes issues with big radii

            // convert to screen units (pixels) for later use
            omega *= screenspaceRadius;

            // line 8 from the paper
            vec3 directionVec = vec3(cosPhi, sinPhi, 0);

            // line 9 from the paper
            vec3 orthoDirectionVec = directionVec - (dot(directionVec, viewVec) * viewVec);

            // line 10 from the paper
            //axisVec is orthogonal to directionVec and viewVec, used to define projectedNormal
            vec3 axisVec = normalize( cross(orthoDirectionVec, viewVec) );

            // line 11 from the paper
            vec3 projectedNormalVec = viewspaceNormal - axisVec * dot(viewspaceNormal, axisVec);

            // line 13 from the paper
            float signNorm = sign( dot( orthoDirectionVec, projectedNormalVec ) );

            // line 14 from the paper
            float projectedNormalVecLength = length(projectedNormalVec);
            float cosNorm = saturate(dot(projectedNormalVec, viewVec) / projectedNormalVecLength);

            // line 15 from the paper
            float n = signNorm * XeGTAO_FastACos(cosNorm);

            // this is a lower weight target; not using -1 as in the original paper because it is under horizon, so a 'weight' has different meaning based on the normal
            float lowHorizonCos0  = cos(n+XE_GTAO_PI_HALF);
            float lowHorizonCos1  = cos(n-XE_GTAO_PI_HALF);


            // lines 17, 18 from the paper, manually unrolled the 'side' loop
            float horizonCos0           = lowHorizonCos0; //-1;
            float horizonCos1           = lowHorizonCos1; //-1;

            // [unroll]
            for( float step = 0; step < stepsPerSlice; step++ )
            {
                // R1 sequence (http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/)
                float stepBaseNoise = float(slice + step * stepsPerSlice) * 0.6180339887498948482; // <- this should unroll
                float stepNoise = fract(noiseSample + stepBaseNoise);

                // approx line 20 from the paper, with added noise
                float s = (step+stepNoise) / (stepsPerSlice); // + (lpfloat2)1e-6f);
                
                // additional distribution modifier
                s       = pow( s, sampleDistributionPower );

                // avoid sampling center pixel
                s       += minS;

                // approx lines 21-22 from the paper, unrolled
                vec2 sampleOffset = s * omega;
                sampleOffset.y = -sampleOffset.y;

                float sampleOffsetLength = length( sampleOffset );

                // note: when sampling, using point_point_point or point_point_linear sampler works, but linear_linear_linear will cause unwanted interpolation between neighbouring depth values on the same MIP level!
                float mipLevel    = clamp( log2( sampleOffsetLength ) - DepthMIPSamplingOffset, 0, XE_GTAO_DEPTH_MIP_LEVELS );

                // Snap to pixel center (more correct direction math, avoids artifacts due to sampling pos not matching depth texel center - messes up slope - but adds other 
                // artifacts due to them being pushed off the slice). Also use full precision for high res cases.
                ivec2 sampleOffsetTexel = ivec2(round(sampleOffset));
                sampleOffset = round(sampleOffset) * ViewportPixelSize;

                vec2 sampleScreenPos0 = normalizedScreenPos + sampleOffset;
                // float SZ0 = texture(ScreenSpaceDepthSampler, sampleScreenPos0).r;
                float SZ0 = textureLod(ScreenSpaceDepthSampler, sampleScreenPos0, mipLevel).r;
                SZ0 = XeGTAO_ScreenSpaceToViewSpaceDepth( SZ0, depthLinearizeMul, depthLinearizeAdd );
                vec3 samplePos0 = XeGTAO_ComputeViewspacePosition( sampleScreenPos0, SZ0, NDCToViewMul, NDCToViewAdd );

                vec2 sampleScreenPos1 = normalizedScreenPos - sampleOffset;
                // float SZ1 = texture(ScreenSpaceDepthSampler, sampleScreenPos1).r;
                float SZ1 = textureLod(ScreenSpaceDepthSampler, sampleScreenPos1, mipLevel).r;
                SZ1 = XeGTAO_ScreenSpaceToViewSpaceDepth( SZ1, depthLinearizeMul, depthLinearizeAdd );
                vec3 samplePos1 = XeGTAO_ComputeViewspacePosition( sampleScreenPos1, SZ1, NDCToViewMul, NDCToViewAdd  );

                vec3 sampleDelta0     = (samplePos0 - pixCenterPos); // using lpfloat for sampleDelta causes precision issues
                vec3 sampleDelta1     = (samplePos1 - pixCenterPos); // using lpfloat for sampleDelta causes precision issues
                float sampleDist0     = length( sampleDelta0 );
                float sampleDist1     = length( sampleDelta1 );

                // approx lines 23, 24 from the paper, unrolled
                vec3 sampleHorizonVec0 = (sampleDelta0 / sampleDist0);
                vec3 sampleHorizonVec1 = (sampleDelta1 / sampleDist1);

                // this is our own thickness heuristic that relies on sooner discarding samples behind the center
                float falloffBase0    = length( vec3(sampleDelta0.x, sampleDelta0.y, sampleDelta0.z * (1+thinOccluderCompensation) ) );
                float falloffBase1    = length( vec3(sampleDelta1.x, sampleDelta1.y, sampleDelta1.z * (1+thinOccluderCompensation) ) );
                float weight0         = saturate( falloffBase0 * falloffMul + falloffAdd );
                float weight1         = saturate( falloffBase1 * falloffMul + falloffAdd );

                // sample horizon cos
                float shc0 = dot(sampleHorizonVec0, viewVec);
                float shc1 = dot(sampleHorizonVec1, viewVec);

                // discard unwanted samples
                shc0 = mix( lowHorizonCos0, shc0, weight0 ); // this would be more correct but too expensive: cos(lerp( acos(lowHorizonCos0), acos(shc0), weight0 ));
                shc1 = mix( lowHorizonCos1, shc1, weight1 ); // this would be more correct but too expensive: cos(lerp( acos(lowHorizonCos1), acos(shc1), weight1 ));


                horizonCos0 = max( horizonCos0, shc0 );
                horizonCos1 = max( horizonCos1, shc1 );

            }

            // I can't figure out the slight overdarkening on high slopes, so I'm adding this fudge - in the training set, 0.05 is close (PSNR 21.34) to disabled (PSNR 21.45)
            projectedNormalVecLength = mix( projectedNormalVecLength, 1, 0.05 );

            // line ~27, unrolled
            float h0 = -XeGTAO_FastACos(horizonCos1);
            float h1 = XeGTAO_FastACos(horizonCos0);


            float iarc0 = (cosNorm + 2.0 * h0 * sin(n)-cos(2.0 * h0-n))/4.0;
            float iarc1 = (cosNorm + 2.0 * h1 * sin(n)-cos(2.0 * h1-n))/4.0;
            float localVisibility = projectedNormalVecLength * (iarc0+iarc1);

            visibility += localVisibility;

        }

        visibility /= sliceCount;
        visibility = pow( visibility, FinalValuePower );
        visibility = max( 0.03, visibility ); // disallow total occlusion (which wouldn't make any sense anyhow since pixel is visible but also helps with packing bent normals)

    }

    XeGTAO_OutputWorkingTerm( visibility, bentNormal, outWorkingAOTerm );
    
}


void XeGTAO_Denoise( const vec2 normUV, const float betaBlurAmount, 
                    const sampler2D aoSampler, const sampler2D edgesSampler, 
                    inout float FinalValue, const bool finalApply)
{
    const float blurAmount = finalApply ? betaBlurAmount : betaBlurAmount / 5.0;
    const float diagWeight = 0.85 * 0.5;

    float aoTerm;
    vec4 edgesC_LRTB;
    float weightTL;
    float weightTR;
    float weightBL;
    float weightBR;

    // GL implementation tweaks / hacks
    // For some reason after porting all this to GL, there was a weird directional ghosting for each pass of the denoiser.
    // I've obviously gotten something wrong, but by introducing this offset the problem went away mostly. There might be
    // some weird very subtle halo'ing around depth discontinuities that may be related to the source problem as well.
    ivec2 UnexplainableSamplingOffset = ivec2(-1,-2);

    // offsets modified from original source for GL
    vec4 edgesQ0        = textureGatherOffset( edgesSampler, normUV, ivec2( 0, 0 )+UnexplainableSamplingOffset, 0 );
    vec4 edgesQ1        = textureGatherOffset( edgesSampler, normUV, ivec2( 2, 0 )+UnexplainableSamplingOffset, 0 );
    vec4 edgesQ2        = textureGatherOffset( edgesSampler, normUV, ivec2( 1, 2 )+UnexplainableSamplingOffset, 0 );

    // offsets modified from original source for GL
    vec4 visQ0 = textureGatherOffset( aoSampler, normUV, ivec2( 0, 2 )+UnexplainableSamplingOffset, 0 ).wzyx;
    vec4 visQ1 = textureGatherOffset( aoSampler, normUV, ivec2( 2, 2 )+UnexplainableSamplingOffset, 0 ).wzyx;
    vec4 visQ2 = textureGatherOffset( aoSampler, normUV, ivec2( 0, 0 )+UnexplainableSamplingOffset, 0 ).wzyx;
    vec4 visQ3 = textureGatherOffset( aoSampler, normUV, ivec2( 2, 0 )+UnexplainableSamplingOffset, 0 ).wzyx;

    vec4 edgesL_LRTB  = XeGTAO_UnpackEdges(edgesQ0.x);
    vec4 edgesT_LRTB  = XeGTAO_UnpackEdges(edgesQ2.w);
    vec4 edgesR_LRTB  = XeGTAO_UnpackEdges(edgesQ1.x);
    vec4 edgesB_LRTB  = XeGTAO_UnpackEdges(edgesQ0.z);

    // DX originals, but still correct due to how the sampling pattern of textureGatherOffset() is laid out.
    edgesC_LRTB    = XeGTAO_UnpackEdges( edgesQ0.y );

    // Edges aren't perfectly symmetrical: edge detection algorithm does not guarantee that a left edge on the right pixel will match the right edge on the left pixel (although
    // they will match in majority of cases). This line further enforces the symmetricity, creating a slightly sharper blur. Works real nice with TAA.
    edgesC_LRTB *= vec4( edgesL_LRTB.y, edgesR_LRTB.x, edgesT_LRTB.w, edgesB_LRTB.z );

   // this allows some small amount of AO leaking from neighbours if there are 3 or 4 edges; this reduces both spatial and temporal aliasing
    const float leak_threshold = 2.5; const float leak_strength = 0.5;
    float edginess = (saturate(4.0 - leak_threshold - dot( edgesC_LRTB, vec4(1.0) )) / (4.0-leak_threshold)) * leak_strength;
    edgesC_LRTB = saturate( edgesC_LRTB + edginess );

    // for diagonals
    weightTL = diagWeight * (edgesC_LRTB.x * edgesL_LRTB.z + edgesC_LRTB.z * edgesT_LRTB.x);
    weightTR = diagWeight * (edgesC_LRTB.z * edgesT_LRTB.y + edgesC_LRTB.y * edgesR_LRTB.z);
    weightBL = diagWeight * (edgesC_LRTB.w * edgesB_LRTB.x + edgesC_LRTB.x * edgesL_LRTB.w);
    weightBR = diagWeight * (edgesC_LRTB.y * edgesR_LRTB.w + edgesC_LRTB.w * edgesB_LRTB.y);

    // first pass
    float ssaoValue     = visQ0[1];
    float ssaoValueL    = visQ0[0];
    float ssaoValueT    = visQ0[2];
    float ssaoValueR    = visQ1[0];
    float ssaoValueB    = visQ2[2];
    float ssaoValueTL   = visQ0[3];
    float ssaoValueBR   = visQ3[3];
    float ssaoValueTR   = visQ1[3];
    float ssaoValueBL   = visQ2[3];

    float sumWeight = blurAmount;
    float sum = ssaoValue * sumWeight;

    XeGTAO_AddSample( ssaoValueL, edgesC_LRTB.x, sum, sumWeight );
    XeGTAO_AddSample( ssaoValueR, edgesC_LRTB.y, sum, sumWeight );
    XeGTAO_AddSample( ssaoValueT, edgesC_LRTB.z, sum, sumWeight );
    XeGTAO_AddSample( ssaoValueB, edgesC_LRTB.w, sum, sumWeight );

    XeGTAO_AddSample( ssaoValueTL, weightTL, sum, sumWeight );
    XeGTAO_AddSample( ssaoValueTR, weightTR, sum, sumWeight );
    XeGTAO_AddSample( ssaoValueBL, weightBL, sum, sumWeight );
    XeGTAO_AddSample( ssaoValueBR, weightBR, sum, sumWeight );

    aoTerm = sum / sumWeight;

    aoTerm *=  (finalApply)?(float(XE_GTAO_OCCLUSION_TERM_SCALE)):(1.0);
    FinalValue = uint(aoTerm * 255.0 + 0.5);

    // read comment above for this function, causes errors so commented out and used directly above.
    // XeGTAO_Output( FinalValue, aoTerm, finalApply );

}