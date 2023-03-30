"""
This extension provides helper utilities for generating uniforms that are frequently passed into the shader.
and also for precomputing some helpful values that we can offload to cpu, and avoid putting to GPU.
"""

import math

class cpuutils:
	"""
	uniformutils description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		# https://github.com/google/filament/blob/main/libs/math/include/math/scalar.h
		self.E                = 2.71828182845904523536028747135266250
		self.LOG2E            = 1.44269504088896340735992468100189214
		self.LOG10E           = 0.434294481903251827651128918916605082
		self.LN2              = 0.693147180559945309417232121458176568
		self.LN10             = 2.30258509299404568401799145468436421
		self.PI               = 3.14159265358979323846264338327950288
		self.PI_2             = 1.57079632679489661923132169163975144
		self.PI_4             = 0.785398163397448309615660845819875721
		self.ONE_OVER_PI      = 0.318309886183790671537767526745028724
		self.TWO_OVER_PI      = 0.636619772367581343075535053490057448
		self.TWO_OVER_SQRTPI  = 1.12837916709551257389615890312154517
		self.SQRT2            = 1.41421356237309504880168872420969808
		self.SQRT1_2          = 0.707106781186547524400844362104849039
		self.TAU              = 2.0 * self.PI
		self.DEG_TO_RAD       = self.PI / 180.0;
		self.RAD_TO_DEG       = 180.0 / self.PI;



	def CameraView_Consts(self, fovX, near, far, renderWidth, renderHeight):
		'''
		extracts and assembles consts related to camera and viewport (render top)

		'''

		ret = {}

		ret['ViewportSize:x'] = renderWidth
		ret['ViewportSize:y'] = renderHeight
		ret['ViewportPixelSize:x'] = 1/renderWidth
		ret['ViewportPixelSize:y'] = 1/renderHeight

		# get the projection matrix from the camera and render top in question.
		projectionMatrix = tdu.Matrix()
		projectionMatrix.projectionFovX(fovX, renderWidth, renderHeight, near, far)

		# gather all the elements of the projection matrix.
		projectionMatrixVals = projectionMatrix.vals

		depthLinearizeMul = -projectionMatrixVals[3 * 4 + 2]
		depthLinearizeAdd = projectionMatrixVals[2 * 4 + 2]

		# line 173-174 https://github.com/GameTechDev/XeGTAO/blob/master/Source/Rendering/Shaders/XeGTAO.h
		if( depthLinearizeMul * depthLinearizeAdd < 0 ):
			depthLinearizeAdd = -depthLinearizeAdd;

		ret['DepthUnpackConsts:mul'] = depthLinearizeMul
		ret['DepthUnpackConsts:add'] = depthLinearizeAdd


		# a matrix is 2d, but it's stored as 1d in memory.
		# this part grabs the appropriate element from the matrix and calculates it's inverse.
		tanHalfFOVY = 1 / projectionMatrixVals[1 + 1 * 4]
		tanHalfFOVX = 1 / projectionMatrixVals[0 + 0 * 4]

		NDCToViewMul = [tanHalfFOVX * 2.0, tanHalfFOVY * 2.0]
		NDCToViewAdd = [tanHalfFOVX * -1.0, tanHalfFOVY * -1.0]

		ret['NDCToViewMul:x'] = NDCToViewMul[0]
		ret['NDCToViewMul:y'] = NDCToViewMul[1]
		ret['NDCToViewAdd:x'] = NDCToViewAdd[0]
		ret['NDCToViewAdd:y'] = NDCToViewAdd[1]

		# consts.NDCToViewMul_x_PixelSize     = { consts.NDCToViewMul.x * consts.ViewportPixelSize.x, consts.NDCToViewMul.y * consts.ViewportPixelSize.y };
		ret['NDCToViewMul_x_PixelSize:x'] = NDCToViewMul[0] * ret['ViewportPixelSize:x']
		ret['NDCToViewMul_x_PixelSize:y'] = NDCToViewMul[1] * ret['ViewportPixelSize:y'] # this might need a -1 mult technically.. but we never seem to use the Y component.

		return ret



	def HilbertIndex(self, posX, posY ):
		'''
		
		additional reference for hilbert curve index function:
		https://github.com/GameTechDev/XeGTAO/blob/master/Source/Rendering/Effects/vaGTAO.cpp
		https://en.wikipedia.org/wiki/Hilbert_curve#Applications_and_mapping_algorithms
		https://dev.to/sandordargo/how-to-use-ampersands-in-c-3kga
		'''

		XE_HILBERT_LEVEL = 6
		XE_HILBERT_WIDTH = 1 << XE_HILBERT_LEVEL # determines width of hilbert texture from number of iterations with bitshifting.
		XE_HILBERT_AREA = XE_HILBERT_WIDTH * XE_HILBERT_WIDTH

		index = 0
		curLevel = XE_HILBERT_WIDTH/2
		while( curLevel > 0 ): # reference function does this with for loop above, but while should work too.
			regionX = ( int(posX) & int(curLevel) ) > 0;
			regionY = ( int(posY) & int(curLevel) ) > 0;
			index += curLevel * curLevel * ( (3 * regionX) ^ regionY)
			index = int(index)
			if( regionY == 0 ):
				if( regionX == 1 ):
					posX = int( (XE_HILBERT_WIDTH - 1) ) - posX
					posY = int( (XE_HILBERT_WIDTH - 1) ) - posY
				temp = int(posX)
				posX = posY
				posY = temp
			curLevel /= 2
			curLevel = int(curLevel)

		return int(index)


	def Srgb2lin(self, s):
		'''
		given an srgb color channel, converts it to a linear color channel.
		'''
		if s <= 0.0404482362771082:
			lin = s / 12.92
		else:
			lin = pow(((s + 0.055) / 1.055), 2.4)
		return lin


	def Lin2srgb(self, lin):
		'''
		given a linear color channel, converts it to an sRGB color channel.
		'''
		if lin > 0.0031308:
			s = 1.055 * (pow(lin, (1.0 / 2.4))) - 0.055
		else:
			s = 12.92 * lin
		return s

	def CalculateLensScaleFactor(self, lensDiam, beamAngle, beamDistance):
		''' 
		all args floats
		'''
		beamAngle = min(beamAngle,90)
		deg2rad = self.PI/180.0
		return 1.0 + 2.0 * math.tan(beamAngle * deg2rad) * beamDistance / lensDiam


	def Rgb_2_Luminance(self,r,g,b):
		'''
		expects r/g/b values as normalized.
		'''
		return (.299 * r) + (.587 * g) + (.114 * b)

	def Mix(self, a, b, mixer):
		return (a * (1-mixer)) + (b * mixer)