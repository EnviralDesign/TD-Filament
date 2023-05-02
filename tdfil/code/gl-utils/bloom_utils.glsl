// Better, temporally stable box filtering
// [Jimenez14] https://goo.gl/eomGso
// . . . . . . .
// . A . B . C .
// . . D . E . .
// . F . G . H .
// . . I . J . .
// . K . L . M .
// . . . . . . .
vec4 DownsampleBox13Tap(sampler2D mipMapSampler, vec2 uv, vec2 texelSize)
{
    vec4 A = texture(mipMapSampler, uv + texelSize * vec2(-1.0, -1.0) );
    vec4 B = texture(mipMapSampler, uv + texelSize * vec2( 0.0, -1.0) );
    vec4 C = texture(mipMapSampler, uv + texelSize * vec2( 1.0, -1.0) );
    vec4 D = texture(mipMapSampler, uv + texelSize * vec2(-0.5, -0.5) );
    vec4 E = texture(mipMapSampler, uv + texelSize * vec2( 0.5, -0.5) );
    vec4 F = texture(mipMapSampler, uv + texelSize * vec2(-1.0,  0.0) );
    vec4 G = texture(mipMapSampler, uv                                );
    vec4 H = texture(mipMapSampler, uv + texelSize * vec2( 1.0,  0.0) );
    vec4 I = texture(mipMapSampler, uv + texelSize * vec2(-0.5,  0.5) );
    vec4 J = texture(mipMapSampler, uv + texelSize * vec2( 0.5,  0.5) );
    vec4 K = texture(mipMapSampler, uv + texelSize * vec2(-1.0,  1.0) );
    vec4 L = texture(mipMapSampler, uv + texelSize * vec2( 0.0,  1.0) );
    vec4 M = texture(mipMapSampler, uv + texelSize * vec2( 1.0,  1.0) );

    vec2 div = (1.0 / 4.0) * vec2(0.5, 0.125);

    vec4 o =  (D + E + I + J) * div.x;
         o += (A + B + G + F) * div.y;
         o += (B + C + H + G) * div.y;
         o += (F + G + L + K) * div.y;
         o += (G + H + M + L) * div.y;

    return o;
}

// Standard box filtering
vec4 DownsampleBox4Tap(sampler2D mipMapSampler, vec2 uv, vec2 texelSize)
{
    vec4 d = texelSize.xyxy * vec4(-1.0, -1.0, 1.0, 1.0);
    vec4 s;

    s =  texture(mipMapSampler , uv + d.xy );
    s += texture(mipMapSampler , uv + d.zy );
    s += texture(mipMapSampler , uv + d.xw );
    s += texture(mipMapSampler , uv + d.zw );

    return s * (1.0 / 4.0);
}

// 9-tap bilinear upsampler (tent filter)
vec4 UpsampleTent( sampler2D mipMapSampler, vec2 uv, vec2 texelSize, vec4 sampleScale )
{
    vec4 d = texelSize.xyxy * vec4(1.0, 1.0, -1.0, 0.0) * sampleScale;

    vec4 s;
    s =  texture(mipMapSampler, uv - d.xy);
    s += texture(mipMapSampler, uv - d.wy) * 2.0;
    s += texture(mipMapSampler, uv - d.zy);

    s += texture(mipMapSampler, uv + d.zw) * 2.0;
    s += texture(mipMapSampler, uv       ) * 4.0;
    s += texture(mipMapSampler, uv + d.xw) * 2.0;

    s += texture(mipMapSampler, uv + d.zy);
    s += texture(mipMapSampler, uv + d.wy) * 2.0;
    s += texture(mipMapSampler, uv + d.xy);

    return s * (1.0 / 16.0);
}

// Standard box filtering
vec4 UpsampleBox( sampler2D mipMapSampler, vec2 uv, vec2 texelSize, vec4 sampleScale )
{
    vec4 d = texelSize.xyxy * vec4(-1.0, -1.0, 1.0, 1.0) * (sampleScale * 0.5);

    vec4 s;
    s =  texture(mipMapSampler, uv + d.xy);
    s += texture(mipMapSampler, uv + d.zy);
    s += texture(mipMapSampler, uv + d.xw);
    s += texture(mipMapSampler, uv + d.zw);

    return s * (1.0 / 4.0);
}