import math
from typing import List, Tuple
import numpy as np
from collections.abc import Iterable

class ColorgradingParams:

	def __init__(self):

		self.toneMapper = self.ACES
		self.hasAdjustments = False
		self.format = "INTEGER"
		self.dimension = 32
		self.luminanceScaling = False
		self.gamutMapping = False
		self.exposure = 0.0
		self.nightAdaptation = 0.0
		self.whiteBalance = [0.0, 0.0]
		self.outRed = [1.0, 0.0, 0.0]
		self.outGreen = [0.0, 1.0, 0.0]
		self.outBlue = [0.0, 0.0, 1.0]
		self.shadows = [1.0, 1.0, 1.0]
		self.midtones = [1.0, 1.0, 1.0]
		self.highlights = [1.0, 1.0, 1.0]
		self.tonalRanges = [0.0, 0.333, 0.550, 1.0]
		self.slope = 1.0
		self.offset = 0.0
		self.power = 1.0
		self.contrast = 1.0
		self.vibrance = 1.0
		self.saturation = 1.0
		self.shadowGamma = 1.0
		self.midPoint = 1.0
		self.highlightScale = 1.0
		self.outputColorSpace = "Rec709-sRGB-D65"
		self.transferFunction = "sRGB"

		# Standard CIE 1931 2° illuminant D65, in xyY space
		self.ILLUMINANT_D65_xyY = [0.31271, 0.32902, 1.0]

		# Standard CIE 1931 2° illuminant D65, in LMS space (CIECAT16)
		# Result of: XYZ_to_CIECAT16 * xyY_to_XYZ(ILLUMINANT_D65_xyY);
		self.ILLUMINANT_D65_LMS_CAT16 = [0.975533, 1.016483, 1.084837]

		# RGB to luminance coefficients for Rec.2020, from Rec2020_to_XYZ
		self.LUMINANCE_Rec2020 = [0.2627002, 0.6779981, 0.0593017]

		# RGB to luminance coefficients for ACEScg (AP1), from AP1_to_XYZ
		self.LUMINANCE_AP1 = [0.272229, 0.674082, 0.0536895]

		# RGB to luminance coefficients for Rec.709, from sRGB_to_XYZ
		self.LUMINANCE_Rec709 = [0.2126730, 0.7151520, 0.0721750]

		# RGB to luminance coefficients for Rec.709 with HK-like weighting
		self.LUMINANCE_HK_Rec709 = [0.13913043, 0.73043478, 0.13043478]

		self.MIDDLE_GRAY_ACEScg = 0.18
		self.MIDDLE_GRAY_ACEScct = 0.4135884

	@property
	def XYZ_to_sRGB(self):
		return [
			[3.2404542, -0.9692660, 0.0556434],
			[-1.5371385, 1.8760108, -0.2040259],
			[-0.4985314, 0.0415560, 1.0572252]
		]

	@property
	def sRGB_to_XYZ(self) -> List[List[float]]:
		return [
			[0.4124560, 0.2126730, 0.0193339],
			[0.3575760, 0.7151520, 0.1191920],
			[0.1804380, 0.0721750, 0.9503040]
		]

	@property
	def Rec2020_to_XYZ(self):
		return [
			[0.6369530, 0.2626983, 0.0000000],
			[0.1446169, 0.6780088, 0.0280731],
			[0.1688558, 0.0592929, 1.0608272],
		]
	
	@property
	def XYZ_to_Rec2020(self):
		return [
			[1.7166634, -0.6666738, 0.0176425],
			[-0.3556733, 1.6164557, -0.0427770],
			[-0.2533681, 0.0157683, 0.9422433]
		]
	
	@property
	def XYZ_to_CIECAT16(self):
		return [
			[0.401288, -0.250268, -0.002079],
			[0.650173, 1.204414, 0.048952],
			[-0.051461, 0.045854, 0.953127]
		]
	
	@property
	def CIECAT16_to_XYZ(self):
		return [
			[1.862068, 0.387527, -0.015841],
			[-1.011255, 0.621447, -0.034123],
			[0.149187, -0.008974, 1.049964]
		]
	
	@property
	def AP1_to_XYZ(self):
		return [
			[0.6624541811, 0.2722287168, -0.0055746495],
			[0.1340042065, 0.6740817658, 0.0040607335],
			[0.1561876870, 0.0536895174, 1.0103391003]
		]
	
	@property
	def XYZ_to_AP1(self):
		return [
			[1.6410233797, -0.6636628587, 0.0117218943],
			[-0.3248032942, 1.6153315917, -0.0082844420],
			[-0.2364246952, 0.0167563477, 0.9883948585]
		]
	
	@property
	def AP1_to_AP0(self):
		return [
			[0.6954522414, 0.0447945634, -0.0055258826],
			[0.1406786965, 0.8596711185, 0.0040252103],
			[0.1638690622, 0.0955343182, 1.0015006723]
		]
	
	@property
	def AP0_to_AP1(self):
		return [
			[1.4514393161, -0.0765537734, 0.0083161484],
			[-0.2365107469, 1.1762296998, -0.0060324498],
			[-0.2149285693, -0.0996759264, 0.9977163014]
		]
	
	@property
	def AP1_to_sRGB(self):
		return [
			[1.70505, -0.13026, -0.02400],
			[-0.62179, 1.14080, -0.12897],
			[-0.08326, -0.01055, 1.15297]
		]
	
	@property
	def sRGB_to_AP1(self):
		return [
			[0.61319, 0.07021, 0.02062],
			[0.33951, 0.91634, 0.10957],
			[0.04737, 0.01345, 0.86961]
		]
	
	@property
	def AP0_to_sRGB(self):
		return [
			[2.52169, -0.27648, -0.01538],
			[-1.13413, 1.37272, -0.15298],
			[-0.38756, -0.09624, 1.16835]
		]
	
	@property
	def sRGB_to_AP0(self):
		return [
			[0.4397010, 0.0897923, 0.0175440],
			[0.3829780, 0.8134230, 0.1115440],
			[0.1773350, 0.0967616, 0.8707040]
		]
	
	@property
	def sRGB_to_OkLab_LMS(self):
		return [
			[0.4122214708, 0.2119034982, 0.0883024619],
			[0.5363325363, 0.6806995451, 0.2817188376],
			[0.0514459929, 0.1073969566, 0.6299787005]
		]
	
	@property
	def XYZ_to_OkLab_LMS(self):
		return [
			[0.8189330101, 0.3618667424, -0.1288597137],
			[0.0329845436, 0.9293118715, 0.0361456387],
			[0.0482003018, 0.2643662691, 0.6338517070]
		]
	
	@property
	def OkLab_LMS_to_XYZ(self):
		return [
			[1.227014, -0.557800, 0.281256],
			[-0.040580, 1.112257, -0.071677],
			[-0.076381, -0.421482, 1.586163]
		]
	
	@property
	def OkLab_LMS_to_OkLab(self):
		return [
			[0.2104542553, 1.9779984951, 0.0259040371],
			[0.7936177850, -2.4285922050, 0.7827717662],
			[-0.0040720468, 0.4505937099, -0.8086757660]
		]
	
	@property
	def OkLab_to_OkLab_LMS(self):
		return [
			[1.0000000000, 1.0000000000, 1.0000000000],
			[0.3963377774, -0.1055613458, -0.0894841775],
			[0.2158037573, -0.0638541728, -1.2914855480]
		]
	
	@property
	def OkLab_LMS_to_sRGB(self):
		return [
			[4.0767416621, -1.2684380046, -0.0041960863],
			[-3.3077115913, 2.6097574011, -0.7034186147],
			[0.2309699292, -0.3413193965, 1.7076147010]
		]

	def mat3(self):
		return [
			[1.0000000000, 0.0000000000, 0.0000000000],
			[0.0000000000, 1.0000000000, 0.0000000000],
			[0.0000000000, 0.0000000000, 1.0000000000]
		]

	@property
	def sRGB_to_Rec2020(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_Rec2020, self.sRGB_to_XYZ)

	@property
	def Rec2020_to_sRGB(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_sRGB, self.Rec2020_to_XYZ)

	@property
	def sRGB_to_LMS_CAT16(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_CIECAT16, self.sRGB_to_XYZ)

	@property
	def LMS_CAT16_to_sRGB(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_sRGB, self.CIECAT16_to_XYZ)

	@property
	def Rec2020_to_LMS_CAT16(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_CIECAT16, self.Rec2020_to_XYZ)

	@property
	def LMS_CAT16_to_Rec2020(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_Rec2020, self.CIECAT16_to_XYZ)

	@property
	def Rec2020_to_AP0(self):
		return self.mat3x3_mul_mat3x3( self.AP1_to_AP0 , self.mat3x3_mul_mat3x3( self.XYZ_to_AP1, self.Rec2020_to_XYZ ) )

	@property
	def AP1_to_Rec2020(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_Rec2020, self.AP1_to_XYZ)

	@property
	def Rec2020_to_OkLab_LMS(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_OkLab_LMS, self.Rec2020_to_XYZ)

	@property
	def OkLab_LMS_to_Rec2020(self):
		return self.mat3x3_mul_mat3x3(self.XYZ_to_Rec2020, self.OkLab_LMS_to_XYZ)

	def sRGB_to_OkLab(self, x):
		x = np.array(x)
		return tuple(self.OkLab_LMS_to_OkLab @ np.cbrt(self.sRGB_to_OkLab_LMS @ x))

	def Rec2020_to_OkLab(self, x):
		x = np.array(x)
		return tuple(self.OkLab_LMS_to_OkLab @ np.cbrt(self.Rec2020_to_OkLab_LMS @ x))

	def OkLab_to_sRGB(self, x):
		x = np.array(x)
		return tuple(self.OkLab_LMS_to_sRGB @ np.power(self.OkLab_to_OkLab_LMS @ x, 3))

	def OkLab_to_Rec2020(self, x):
		x = np.array(x)
		return tuple(self.OkLab_LMS_to_Rec2020 @ np.power(self.OkLab_to_OkLab_LMS @ x, 3))

	@property
	def selectColorGradingTransformIn(self):
		if self.toneMapper == self.FILMIC:
			return self.mat3
		return self.sRGB_to_Rec2020

	@property
	def selectColorGradingTransformOut(self):
		if self.toneMapper == self.FILMIC:
			return self.mat3
		return self.Rec2020_to_sRGB

	@property
	def selectColorGradingLuminance(self):
		if self.toneMapper == self.FILMIC:
			return self.LUMINANCE_Rec709
		return self.LUMINANCE_Rec2020
	
	@property
	def selectOETF(self):
		if self.transferFunction == 'linear':
			return self.OETF_Linear
		return self.OETF_sRGB

	def OETF_Linear(self, x):
		return x

	def OETF_sRGB(self, x):
		x = np.array(x)
		a = 0.055
		a1 = 1.055
		b = 12.92
		p = 1 / 2.4
		x = np.where(x <= 0.0031308, x * b, a1 * np.power(x, p) - a)
		return x

	def mat3x3_mul_vec3(self, mat: List[List[float]], vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
		x, y, z = vec
		return (
			mat[0][0] * x + mat[0][1] * y + mat[0][2] * z,
			mat[1][0] * x + mat[1][1] * y + mat[1][2] * z,
			mat[2][0] * x + mat[2][1] * y + mat[2][2] * z
		)

	def mat3x3_mul_mat3x3(self, mat1: List[List[float]], mat2: List[List[float]]) -> List[List[float]]:
		result = [[0.0 for _ in range(3)] for _ in range(3)]
		for i in range(3):
			for j in range(3):
				for k in range(3):
					result[i][j] += mat1[i][k] * mat2[k][j]
		return result
	
	def vec3_mul_vec3(self, vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> Tuple[float, float, float]:
		x1, y1, z1 = vec1
		x2, y2, z2 = vec2
		return (x1 * x2, y1 * y2, z1 * z2)

	def vec3_div_vec3(self, vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> Tuple[float, float, float]:
		x1, y1, z1 = vec1
		x2, y2, z2 = vec2
		return (x1 / x2, y1 / y2, z1 / z2)
	
	def vec3_mul_float(self, vec: Tuple[float, float, float], scalar: float) -> Tuple[float, float, float]:
		x, y, z = vec
		return (x * scalar, y * scalar, z * scalar)
	
	def clamp_float3(self, vec: Tuple[float, float, float], low: float, high: float) -> Tuple[float, float, float]:
		x, y, z = vec
		return (
			max(min(x, high), low),
			max(min(y, high), low),
			max(min(z, high), low)
		)
	
	def dot_product(self, vec1: Tuple[float, float, float], vec2: Tuple[float, float, float]) -> float:
		x1, y1, z1 = vec1
		x2, y2, z2 = vec2
		return x1 * x2 + y1 * y2 + z1 * z2
	
	def mix_float3(self, vec1: Tuple[float, float, float], vec2: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
		return (
			vec1[0] * (1.0 - t) + vec2[0] * t,
			vec1[1] * (1.0 - t) + vec2[1] * t,
			vec1[2] * (1.0 - t) + vec2[2] * t
		)
	
	def saturate_1(self, v: float) -> float:
		return max(min(v, 1), 0)
	
	def saturate_n(self, x: List[float]) -> List[float]:
		return [max(min(xi, 1), 0) for xi in x]

	def cbrt(self, v: List[float]) -> List[float]:
		# cube root
		for i in range(len(v)):
			v[i] = pow(v[i], 1/3)
		return v
	
	def smoothstep(self, edge0: float, edge1: float, v: float) -> float:
		return max(min((v - edge0) / (edge1 - edge0), 1), 0) ** 2 * (3 - 2 * max(min((v - edge0) / (edge1 - edge0), 1), 0))

	def smoothstep_list(self, edge0: float, edge1: float, v: List[float]) -> List[float]:
		def saturate(x: List[float]) -> List[float]:
			return [max(min(xi, 1), 0) for xi in x]
		
		t = saturate([(vi - edge0) / (edge1 - edge0) for vi in v])
		return [ti * ti * (3 - 2 * ti) for ti in t]

	def chromaticityCoordinateIlluminantD(self, x: float) -> float:
		return 2.87 * x - 3.0 * x * x - 0.275
	
	def xyY_to_XYZ(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
		a = v[2] / max(v[1], 1e-5)
		return (v[0] * a, v[2], (1.0 - v[0] - v[1]) * a)

	def XYZ_to_xyY(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
		divisor = max(sum(v), 1e-5)
		return (v[0] / divisor, v[1] / divisor, v[1])

	def pow3(self, x: Tuple[float, float, float]) -> Tuple[float, float, float]:
		return (x[0] * x[0] * x[0], x[1] * x[1] * x[1], x[2] * x[2] * x[2])

	def sgn(self, x):
		return (0.0 < x).astype(int) ^ (x < 0.0).astype(int)
	
	def compute_max_saturation(self, a, b):
		# Max saturation will be when one of r, g, or b goes below zero.

		# Select different coefficients depending on which component goes below zero first
		if (-1.88170328 * a - 0.80936493 * b > 1):
			# Red component
			k0, k1, k2, k3, k4 = 1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245
			wl, wm, ws = 4.0767416621, -3.3077115913, 0.2309699292
		elif (1.81444104 * a - 1.19445276 * b > 1):
			# Green component
			k0, k1, k2, k3, k4 = 0.73956515, -0.45954404, 0.08285427, 0.12541070, 0.14503204
			wl, wm, ws = -1.2681437731, 2.6097574011, -0.3413193965
		else:
			# Blue component
			k0, k1, k2, k3, k4 = 1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167
			wl, wm, ws = -0.0041960863, -0.7034186147, 1.7076147010

		# Approximate max saturation using a polynomial:
		S = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b

		# Do one step Halley's method to get closer
		k_l = 0.3963377774 * a + 0.2158037573 * b
		k_m = -0.1055613458 * a - 0.0638541728 * b
		k_s = -0.0894841775 * a - 1.2914855480 * b

		l_ = 1.0 + S * k_l
		m_ = 1.0 + S * k_m
		s_ = 1.0 + S * k_s

		l = l_ * l_ * l_
		m = m_ * m_ * m_
		s = s_ * s_ * s_

		l_dS = 3.0 * k_l * l_ * l_
		m_dS = 3.0 * k_m * m_ * m_
		s_dS = 3.0 * k_s * s_ * s_

		l_dS2 = 6.0 * k_l * k_l * l_
		m_dS2 = 6.0 * k_m * k_m * m_
		s_dS2 = 6.0 * k_s * k_s * s_

		f = wl * l + wm * m + ws * s
		f1 = wl * l_dS + wm * m_dS + ws * s_dS
		f2 = wl * l_dS2 + wm * m_dS2 + ws * s_dS2

		S = S - f * f1 / (f1 * f1 - 0.5 * f * f2)

		return S
	
	
	def find_cusp(self, a, b):
		# First, find the maximum saturation (saturation S = C/L)
		S_cusp = self.compute_max_saturation(a, b)

		# Convert to linear sRGB to find the first point where at least one of r,g or b >= 1:
		rgb_at_max = self.OkLab_to_sRGB([1.0, S_cusp * a, S_cusp * b])
		L_cusp = np.cbrt(1.0 / np.max(rgb_at_max))
		C_cusp = L_cusp * S_cusp

		return [L_cusp, C_cusp]
	
	def find_gamut_intersection(self, a, b, L1, C1, L0):
		# Assume find_cusp is defined elsewhere and returns a tuple (x, y)
		cusp = self.find_cusp(a, b)

		if ((L1 - L0) * cusp[1] - (cusp[0] - L0) * C1) <= 0.0:
			# Lower half
			t = cusp[1] * L0 / (C1 * cusp[0] + cusp[1] * (L0 - L1))
		else:
			# Upper half
			t = cusp[1] * (L0 - 1.0) / (C1 * (cusp[0] - 1.0) + cusp[1] * (L0 - L1))

			dL = L1 - L0
			dC = C1

			k_l = +0.3963377774 * a + 0.2158037573 * b
			k_m = -0.1055613458 * a - 0.0638541728 * b
			k_s = -0.0894841775 * a - 1.2914855480 * b

			l_dt = dL + dC * k_l
			m_dt = dL + dC * k_m
			s_dt = dL + dC * k_s

			L = L0 * (1.0 - t) + t * L1
			C = t * C1

			l_ = L + C * k_l
			m_ = L + C * k_m
			s_ = L + C * k_s

			l = l_ * l_ * l_
			m = m_ * m_ * m_
			s = s_ * s_ * s_

			ldt = 3 * l_dt * l_ * l_
			mdt = 3 * m_dt * m_ * m_
			sdt = 3 * s_dt * s_ * s_

			ldt2 = 6 * l_dt * l_dt * l_
			mdt2 = 6 * m_dt * m_dt * m_
			sdt2 = 6 * s_dt * s_dt * s_

			r = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s - 1
			r1 = 4.0767416621 * ldt - 3.3077115913 * mdt + 0.2309699292 * sdt
			r2 = 4.0767416621 * ldt2 - 3.3077115913 * mdt2 + 0.2309699292 * sdt2

			u_r = r1 / (r1 * r1 - 0.5 * r * r2)
			t_r = -r * u_r

			g = -1.2681437731 * l + 2.6097574011 * m - 0.3413193965 * s - 1
			g1 = -1.2681437731 * ldt + 2.6097574011 * mdt - 0.3413193965 * sdt
			g2 = -1.2681437731 * ldt2 + 2.6097574011 * mdt2 - 0.3413193965 * sdt2

			u_g = g1 / (g1 * g1 - 0.5 * g * g2)
			t_g = -g * u_g

			b0 = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s - 1
			b1 = -0.0041960863 * ldt - 0.7034186147 * mdt + 1.7076147010 * sdt
			b2 = -0.0041960863 * ldt2 - 0.7034186147 * mdt2 + 1.7076147010 * sdt2

			u_b = b1 / (b1 * b1 - 0.5 * b0 * b2)
			t_b = -b0 * u_b

			t_r = t_r if u_r >= 0.0 else float('inf')
			t_g = t_g if u_g >= 0.0 else float('inf')
			t_b = t_b if u_b >= 0.0 else float('inf')

			t += min(t_r, min(t_g, t_b))

		return t



	def gamut_clip_adaptive_L0_0_5(self, rgb, alpha=0.05, threshold=0.03):
		rgb = np.array(rgb)
		if np.all(np.logical_and(rgb <= 1.0 + threshold, rgb >= -threshold)):
			return tuple(rgb)

		lab = self.sRGB_to_OkLab(rgb)

		L = lab[0]
		eps = 0.00001
		C = max(eps, np.sqrt(lab[1] * lab[1] + lab[2] * lab[2]))
		a_ = lab[1] / C
		b_ = lab[2] / C

		Ld = L - 0.5
		e1 = 0.5 + np.abs(Ld) + alpha * C
		L0 = 0.5 * (1.0 + self.sgn(Ld) * (e1 - np.sqrt(e1 * e1 - 2.0 * np.abs(Ld))))

		t = self.find_gamut_intersection(a_, b_, L, C, L0)

		L_clipped = L0 * (1.0 - t) + t * L
		C_clipped = t * C

		return tuple(self.OkLab_to_sRGB((L_clipped, C_clipped * a_, C_clipped * b_)))

	
	def gamut_mapping_sRGB(self, rgb):
		return self.gamut_clip_adaptive_L0_0_5(rgb)


	def rgb_2_saturation(self, rgb: Tuple[float, float, float]) -> float:
		# Input: ACES
		# Output: OCES
		TINY = 1e-5
		mi = min(rgb)
		ma = max(rgb)
		return (max(ma, TINY) - max(mi, TINY)) / max(ma, 1e-2)
	
	def rgb_2_yc(self, rgb: Tuple[float, float, float], ycRadiusWeight: float = 1.75) -> float:
		# Converts RGB to a luminance proxy, here called YC
		# YC is ~ Y + K * Chroma
		# Constant YC is a cone-shaped surface in RGB space, with the tip on the
		# neutral axis, towards white.
		# YC is normalized: RGB 1 1 1 maps to YC = 1
		#
		# ycRadiusWeight defaults to 1.75, although can be overridden in function
		# call to rgb_2_yc
		# ycRadiusWeight = 1 -> YC for pure cyan, magenta, yellow == YC for neutral
		# of same value
		# ycRadiusWeight = 2 -> YC for pure red, green, blue  == YC for  neutral of
		# same value.

		r, g, b = rgb

		chroma = math.sqrt(b * (b - g) + g * (g - r) + r * (r - b))

		return (b + g + r + ycRadiusWeight * chroma) / 3.0

	def sigmoid_shaper(self, x: float) -> float:
		# Sigmoid function in the range 0 to 1 spanning -2 to +2.
		t = max(1.0 - abs(x / 2.0), 0.0)
		y = 1.0 + math.copysign(1.0, x) * (1.0 - t * t)
		return y / 2.0
	
	def glow_fwd(self, ycIn: float, glowGainIn: float, glowMid: float) -> float:
		# Compute the glow gain output based on the input luminance proxy value.
		# The glow gain output is zero for input values greater than or equal to
		# 2.0 * glowMid, equal to glowGainIn for input values less than or equal to
		# 2.0 / 3.0 * glowMid, and a sigmoid function of the input value for values
		# in between those two thresholds.

		if ycIn <= 2.0 / 3.0 * glowMid:
			glowGainOut = glowGainIn
		elif ycIn >= 2.0 * glowMid:
			glowGainOut = 0.0
		else:
			t = max(1.0 - abs((ycIn / glowMid) - 1.5), 0.0)
			glowGainOut = glowGainIn * t

		return glowGainOut
	

	def linear_to_LogC(self, x):
		a = 5.555556
		b = 0.047996
		c = 0.244161
		d = 0.386036
		return tuple(c * math.log10(a * value + b) + d for value in x)

	
	
	def scotopicAdaptation(self, v: Tuple[float, float, float], nightAdaptation: float) -> Tuple[float, float, float]:
		'''
		//------------------------------------------------------------------------------
		// Purkinje shift/scotopic vision
		//------------------------------------------------------------------------------

		// In low-light conditions, peak luminance sensitivity of the eye shifts toward
		// the blue end of the visible spectrum. This effect called the Purkinje effect
		// occurs during the transition from photopic (cone-based) vision to scotopic
		// (rod-based) vision. Because the rods and cones use the same neural pathways,
		// a color shift is introduced as the rods take over to improve low-light
		// perception.
		//
		// This function aims to (somewhat) replicate this color shift and peak luminance
		// sensitivity increase to more faithfully reproduce scenes in low-light conditions
		// as they would be perceived by a human observer (as opposed to an artificial
		// observer such as a camera sensor).
		//
		// The implementation below is based on two papers:
		// "Rod Contributions to Color Perception: Linear with Rod Contrast", Cao et al., 2008
		//     https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2630540/pdf/nihms80286.pdf
		// "Perceptually Based Tone Mapping for Low-Light Conditions", Kirk & O'Brien, 2011
		//     http://graphics.berkeley.edu/papers/Kirk-PBT-2011-08/Kirk-PBT-2011-08.pdf
		//
		// Many thanks to Jasmin Patry for his explanations in "Real-Time Samurai Cinema",
		// SIGGRAPH 2021, and the idea of using log-luminance based on "Maximum Entropy
		// Spectral Modeling Approach to Mesopic Tone Mapping", Rezagholizadeh & Clark, 2013
		
		// The 4 vectors below are generated by the command line tool rgb-to-lmsr.
		// Together they form a 4x3 matrix that can be used to convert a Rec.709
		// input color to the LMSR (long/medium/short cone + rod receptors) space.
		// That matrix is computed using this formula:
		//     Mij = \Integral Ei(lambda) I(lambda) Rj(lambda) d(lambda)
		// Where:
		//     i in {L, M, S, R}
		//     j in {R, G, B}
		//     lambda: wavelength
		//     Ei(lambda): response curve of the corresponding receptor
		//     I(lambda): relative spectral power of the CIE illuminant D65
		//     Rj(lambda): spectral power of the corresponding Rec.709 color
		'''

		v = np.array(v)

		# constexpr float3 L{7.696847f, 18.424824f,  2.068096f};
		# constexpr float3 M{2.431137f, 18.697937f,  3.012463f};
		# constexpr float3 S{0.289117f,  1.401833f, 13.792292f};
		# constexpr float3 R{0.466386f, 15.564362f, 10.059963f};
		L = (7.696847, 18.424824, 2.068096)
		M = (2.431137, 18.697937, 3.012463)
		S = (0.289117, 1.401833, 13.792292)
		R = (0.466386, 15.564362, 10.059963)

		# constexpr mat3f LMS_to_RGB = inverse(transpose(mat3f{L, M, S}));
		LMS_to_RGB = np.linalg.inv(np.transpose(np.array([L, M, S])))

		# // Maximal LMS cone sensitivity, Cao et al. Table 1
		# constexpr float3 m{0.63721f, 0.39242f, 1.6064f};
		m = np.array([0.63721, 0.39242, 1.6064])
		# // Strength of rod input, free parameters in Cao et al., manually tuned for our needs
		# // We follow Kirk & O'Brien who recommend constant values as opposed to Cao et al.
		# // who propose to adapt those values based on retinal illuminance. We instead offer
		# // artistic control at the end of the process
		# // The vector below is {k1, k1, k2} in Kirk & O'Brien, but {k5, k5, k6} in Cao et al.
		# constexpr float3 k{0.2f, 0.2f, 0.3f};
		k = np.array([0.2, 0.2, 0.3])
		
		# // Transform from opponent space back to LMS
		# constexpr mat3f opponent_to_LMS{
		# 	-0.5f, 0.5f, 0.0f,
		# 	0.0f, 0.0f, 1.0f,
		# 	0.5f, 0.5f, 1.0f
		# };
		opponent_to_LMS = np.array([[-0.5, 0.5, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 1.0]])

		# // The constants below follow Cao et al, using the KC pathway
		# // Scaling constant
		# constexpr float K_ = 45.0f;
		# // Static saturation
		# constexpr float S_ = 10.0f;
		# // Surround strength of opponent signal
		# constexpr float k3 = 0.6f;
		# // Radio of responses for white light
		# constexpr float rw = 0.139f;
		# // Relative weight of L cones
		# constexpr float p  = 0.6189f;
		K_ = 45.0
		S_ = 10.0
		k3 = 0.6
		rw = 0.139
		p = 0.6189

		# // Weighted cone response as described in Cao et al., section 3.3
		# // The approximately linear relation defined in the paper is represented here
		# // in matrix form to simplify the code
		# constexpr mat3f weightedRodResponse = (K_ / S_) * mat3f{
		# -(k3 + rw),       p * k3,          p * S_,
		# 	1.0f + k3 * rw, (1.0f - p) * k3, (1.0f - p) * S_,
		# 	0.0f,            1.0f,            0.0f
		# } * mat3f{k} * inverse(mat3f{m});
		weightedRodResponse = (K_ / S_) * \
		np.array(
			[[-(k3 + rw), p * k3, p * S_], 
    		[1.0 + k3 * rw, (1.0 - p) * k3, (1.0 - p) * S_], 
			[0.0, 1.0, 0.0]]
			) * np.array([k]) * np.linalg.inv(np.diag(m))
		
		# // Move to log-luminance, or the EV values as measured by a Minolta Spotmeter F.
		# // The relationship is EV = log2(L * 100 / 14), or 2^EV = L / 0.14. We can therefore
		# // multiply our input by 0.14 to obtain our log-luminance values.
		# // We then follow Patry's recommendation to shift the log-luminance by ~ +11.4EV to
		# // match luminance values to mesopic measurements as described in Rezagholizadeh &
		# // Clark 2013,
		# // The result is 0.14 * exp2(11.40) ~= 380.0 (we use +11.406 EV to get a round number)
		# constexpr float logExposure = 380.0f;
		logExposure = 380.0

		# // Move to scaled log-luminance
		v *= logExposure

		# // Convert the scene color from Rec.709 to LMSR response
		# float4 q{dot(v, L), dot(v, M), dot(v, S), dot(v, R)};
		q = np.array([np.dot(v, L), np.dot(v, M), np.dot(v, S), np.dot(v, R)])

		# // Regulated signal through the selected pathway (KC in Cao et al.)
		# float3 g = inversesqrt(1.0f + max(float3{0.0f}, (0.33f / m) * (q.rgb + k * q.w)));
		g = np.sqrt(1.0 / (1.0 + np.maximum(0.0, (0.33 / m) * (q[0:3] + k * q[3]))))

		# // Compute the incremental effect that rods have in opponent space
		# float3 deltaOpponent = weightedRodResponse * g * q.w * nightAdaptation;
		deltaOpponent = np.dot(weightedRodResponse, g * q[3] * nightAdaptation)
		# // Photopic response in LMS space
		# float3 qHat = q.rgb + opponent_to_LMS * deltaOpponent;
		qHat = q[0:3] + np.dot(opponent_to_LMS, deltaOpponent)

		# // And finally, back to RGB
		# return (LMS_to_RGB * qHat) / logExposure;
		# return np.dot(LMS_to_RGB, qHat) / logExposure
		ret = np.dot(LMS_to_RGB, qHat) / logExposure
		ret = list(ret)
		ret = tuple(ret)
		return ret

	def adaptationTransform(self, whiteBalance: Tuple[float, float]) -> Tuple[float, float]:
		'''
		# // Return the chromatic adaptation coefficients in LMS space for the given
	# // temperature/tint offsets. The chromatic adaption is perfomed following
	# // the von Kries method, using the CIECAT16 transform.
	# // See https://en.wikipedia.org/wiki/Chromatic_adaptation
	# // See https://en.wikipedia.org/wiki/CIECAM02#Chromatic_adaptation
		'''
		# See Mathematica notebook in docs/math/White Balance.nb
		k = whiteBalance[0] # temperature
		t = whiteBalance[1] # tint

		x = self.ILLUMINANT_D65_xyY[0] - k * (k < 0.0 and 0.0214 or 0.066)
		y = self.chromaticityCoordinateIlluminantD(x) + t * 0.066

		lms = np.dot(self.XYZ_to_CIECAT16, self.xyY_to_XYZ([x, y, 1.0]))
		ret = np.dot(self.LMS_CAT16_to_Rec2020, np.dot(np.diag(self.ILLUMINANT_D65_LMS_CAT16 / lms), self.Rec2020_to_LMS_CAT16))
		return ret
	
	def chromaticAdaptation(self, v: Tuple[float, float, float], adaptationTransform: np.array) -> Tuple[float, float, float]:
		# return adaptationTransform * v;
		# print(adaptationTransform)
		return adaptationTransform @ np.array(v)

	def rgb_2_hue(self, rgb: Tuple[float, float, float]) -> float:
		# Returns a geometric hue angle in degrees (0-360) based on RGB values.
		# For neutral colors, hue is undefined and the function will return a quiet NaN value.
		hue = 0.0
		# RGB triplets where RGB are equal have an undefined hue
		if not (rgb[0] == rgb[1] and rgb[1] == rgb[2]):
			hue = math.degrees(math.atan2(math.sqrt(3.0) * (rgb[1] - rgb[2]), 2.0 * rgb[0] - rgb[1] - rgb[2]))
		return hue if hue >= 0.0 else hue + 360.0

	def center_hue(self, hue: float, centerH: float) -> float:
		# Centers the hue value around a specified center hue value.
		hue_centered = hue - centerH
		if hue_centered < -180.0:
			hue_centered += 360.0
		elif hue_centered > 180.0:
			hue_centered -= 360.0
		return hue_centered
	
	def darkSurround_to_dimSurround(self, linearCV: Tuple[float, float, float]) -> Tuple[float, float, float]:
		# Convert linear color values in a dark surround to a dim surround.
		DIM_SURROUND_GAMMA = 0.9811

		# XYZ = self.AP1_to_XYZ(linearCV)
		XYZ = self.mat3x3_mul_vec3(self.AP1_to_XYZ, linearCV)
		xyY = self.XYZ_to_xyY(XYZ)
		xyY = list(xyY)

		xyY_z_clamped = max(0.0, min(xyY[2], math.pow(math.pow(2, 15) - 1, 2.0)))
		xyY_z_dim_surround = math.pow(xyY_z_clamped, DIM_SURROUND_GAMMA)

		xyY[2] = xyY_z_dim_surround

		XYZ = self.xyY_to_XYZ(xyY)
		return self.mat3x3_mul_vec3(self.XYZ_to_AP1, XYZ)

	def LogC_to_linear(self, x):
		### expects numpy array!
		x = np.array(x)
		ia = 1.0 / 5.555556
		b = 0.047996
		ic = 1.0 / 0.244161
		d = 0.386036
		return (np.power(10.0, (x - d) * ic) - b) * ia

	def luminanceScaling_(self, x, toneMapper, luminanceWeights):
		# Troy Sobotka, 2021, "EVILS - Exposure Value Invariant Luminance Scaling"
		# https://colab.research.google.com/drive/1iPJzNNKR7PynFmsqSnQm3bCZmQ3CvAJ-#scrollTo=psU43hb-BLzB

		luminanceIn = np.dot(x, luminanceWeights)
		# print(luminanceIn)

		# TODO: We could optimize for the case of single-channel luminance
		luminanceOut = toneMapper(luminanceIn, 1)[1]

		peak = np.max(x)
		chromaRatio = np.maximum(x / peak, 0.0)

		chromaRatioLuminance = np.dot(chromaRatio, luminanceWeights)

		maxReserves = 1.0 - chromaRatio
		maxReservesLuminance = np.dot(maxReserves, luminanceWeights)

		luminanceDifference = np.maximum(luminanceOut - chromaRatioLuminance, 0.0)
		scaledLuminanceDifference = luminanceDifference / np.maximum(maxReservesLuminance, np.finfo(float).tiny)

		chromaScale = (luminanceOut - luminanceDifference) / np.maximum(chromaRatioLuminance, np.finfo(float).tiny)

		return chromaScale * chromaRatio + scaledLuminanceDifference * maxReserves

	def adjust_exposure(self, v, exposure):
		return tuple(x * pow(2, exposure) for x in v)

	def is_iterable(self, obj):
		return isinstance(obj, Iterable)

	def ACES(self, color: Tuple[float, float, float], brightness: float) -> Tuple[float, float, float]:
		# Some bits were removed to adapt to our desired output

		# hack to ensure passed in args are actually a float3.
		is_iterable = self.is_iterable(color)
		if is_iterable == False:
			color = (color, color, color)
		else:
			if len(color) != 3:
				raise ValueError("color must be a float3")

		# "Glow" module constants
		RRT_GLOW_GAIN = 0.05
		RRT_GLOW_MID = 0.08

		# Red modifier constants
		RRT_RED_SCALE = 0.82
		RRT_RED_PIVOT = 0.03
		RRT_RED_HUE = 0.0
		RRT_RED_WIDTH = 135.0

		# Desaturation constants
		RRT_SAT_FACTOR = 0.96
		ODT_SAT_FACTOR = 0.93

		ap0 = self.mat3x3_mul_vec3(self.Rec2020_to_AP0, color)
		# ap0 = mul(Rec2020_to_AP0, color)

		# Glow module
		saturation = self.rgb_2_saturation(ap0)
		ycIn = self.rgb_2_yc(ap0)
		s = self.sigmoid_shaper((saturation - 0.4) / 0.2)
		addedGlow = 1.0 + self.glow_fwd(ycIn, RRT_GLOW_GAIN * s, RRT_GLOW_MID)
		# print(addedGlow)
		ap0 = list(self.vec3_mul_float(ap0, addedGlow))

		# Red modifier
		hue = self.rgb_2_hue(ap0)
		centeredHue = self.center_hue(hue, RRT_RED_HUE)
		# hueWeight = self.smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centeredHue / RRT_RED_WIDTH))
		hueWeight = self.smoothstep(0.0, 1.0, abs(2.0 * centeredHue / RRT_RED_WIDTH))
		hueWeight *= hueWeight
		# hueWeight *= hueWeight

		ap0[0] += hueWeight * saturation * (RRT_RED_PIVOT - ap0[0]) * (1.0 - RRT_RED_SCALE)

		# scalar = ((RRT_RED_PIVOT - ap0[0]) * (1.0 - RRT_RED_SCALE))
		# otherscalar = hueWeight * saturation
		# ap0[0] += otherscalar * scalar

		# ACES to RGB rendering space
		# ap1 = clamp(mul(AP0_to_AP1, ap0), 0.0, float('inf'))
		ap1 = self.clamp_float3(self.mat3x3_mul_vec3(self.AP0_to_AP1, ap0), 0.0, float('inf'))

		# Global desaturation
		# ap1 = mix(dot(ap1, LUMINANCE_AP1), ap1, RRT_SAT_FACTOR)
		# print(self.dot_product(ap1, self.LUMINANCE_AP1))
		mix0 = [self.dot_product(ap1, self.LUMINANCE_AP1)]*3
		mix1 = ap1
		ap1 = self.mix_float3(mix0, mix1, RRT_SAT_FACTOR)

		# NOTE: This is specific to Filament and added only to match ACES to our legacy tone mapper
		#       which was a fit of ACES in Rec.709 but with a brightness boost.
		ap1 = self.vec3_mul_float(ap1, brightness)

		# Fitting of RRT + ODT (RGB monitor 100 nits dim) from:
		# https://github.com/colour-science/colour-unity/blob/master/Assets/Colour/Notebooks/CIECAM02_Unity.ipynb
		a = 2.785085
		b = 0.107772
		c = 2.936045
		d = 0.887122
		e = 0.806889

		ap1 = np.array(ap1)
		# float3 rgbPost = (ap1 * (a * ap1 + b)) / (ap1 * (c * ap1 + d) + e);
		rgbPost = (ap1 * (a * ap1 + b)) / (ap1 * (c * ap1 + d) + e)

		# Apply gamma adjustment to compensate for dim surround
		linearCV = self.darkSurround_to_dimSurround(rgbPost)

		# Apply desaturation to compensate for luminance difference
		# linearCV = mix(dot(linearCV, LUMINANCE_AP1), linearCV, ODT_SAT_FACTOR)
		# print(linearCV, self.LUMINANCE_AP1)
		mix1 = [self.dot_product(linearCV, self.LUMINANCE_AP1)]*3
		mix2 = linearCV
		linearCV = self.mix_float3(mix1, mix2, ODT_SAT_FACTOR)

		# return mul(AP1_to_Rec2020, linearCV)
		return self.mat3x3_mul_vec3(self.AP1_to_Rec2020, linearCV)
	
	def FILMIC(self, color: Tuple[float, float, float] ) -> Tuple[float, float, float]:
		# Narkowicz 2015, "ACES Filmic Tone Mapping Curve"
		a = 2.51
		b = 0.03
		c = 2.43
		d = 0.59
		e = 0.14
		r_, g_, b_ = color
		r = (r_ * (a * r_ + b)) / (r_ * (c * r_ + d) + e)
		g = (g_ * (a * g_ + b)) / (g_ * (c * g_ + d) + e)
		b = (b_ * (a * b_ + b)) / (b_ * (c * b_ + d) + e)
		return [r, g, b]
	
	
	def Set_format(self, format):
		'''
		format: 'INTEGER', 'FLOAT'
		'''
		assert format in ['INTEGER', 'FLOAT'] , "format must be 'INTEGER' or 'FLOAT'"
		self.format = format
		return

	def Set_dimension(self, dimension):
		'''
		dimension: 16, 32, 64
		'''
		assert dimension in [16, 32, 64] , "dimension must be 16, 32, or 64"
		self.dimension = dimension
		return

	def Set_luminanceScaling(self, luminanceScaling):
		'''
		luminanceScaling: True, False
		'''
		assert luminanceScaling in [True, False] , "luminanceScaling must be True or False"
		self.luminanceScaling = luminanceScaling
		return

	def Set_gamutMapping(self, gamutMapping):
		'''
		gamutMapping: True, False
		'''
		assert gamutMapping in [True, False] , "gamutMapping must be True or False"
		self.gamutMapping = gamutMapping
		return
	
	def Set_toneMapper(self, toneMapper):
		'''
		toneMapper: 'ACES'
		'''
		assert toneMapper in ['ACES'] , "toneMapper must be 'ACES', 'FILMIC'"
		if toneMapper == 'ACES':
			self.toneMapper = self.ACES
		elif toneMapper == 'FILMIC':
			self.toneMapper = self.FILMIC
		
		return
	
	def Set_exposure(self, exposure):
		'''
		exposure: float
		'''
		self.exposure = exposure
		return

	def Set_nightAdaptation(self, nightAdaptation):
		'''
		nightAdaptation: float
		'''
		self.nightAdaptation = nightAdaptation
		return
	
	def Set_whiteBalance(self, temperature, tint):
		'''
		whiteBalance: [float, float]
		'''
		temperature = math.clamp(temperature, -1.0, 1.0)
		tint = math.clamp(tint, -1.0, 1.0)
		self.whiteBalance = [temperature, tint]
		return
	
	def Set_channelMixer(self, outRed, outGreen, outBlue):
		'''
		channelMixer: [float, float, float], [float, float, float], [float, float, float]
		'''
		self.outRed = math.clamp(outRed, -2, 2)
		self.outGreen = math.clamp(outGreen, -2, 2)
		self.outBlue = math.clamp(outBlue, -2, 2)
		return
	
	def set_shadowsMidtonesHighlights(self, shadows4, midtones4, highlights4):
		
		assert len(shadows4) == 4, "shadows4 must be a list of 4 floats"
		assert len(midtones4) == 4, "midtones4 must be a list of 4 floats"
		assert len(highlights4) == 4, "highlights4 must be a list of 4 floats"

		shadows3 = [
			max( shadows4[0] + shadows4[3] , 0 ), # shadows r
			max( shadows4[1] + shadows4[3] , 0 ), # shadows g
			max( shadows4[2] + shadows4[3] , 0 ) # shadows b
		]

		midtones3 = [
			max( midtones4[0] + midtones4[3] , 0 ), # midtones r
			max( midtones4[1] + midtones4[3] , 0 ), # midtones g
			max( midtones4[2] + midtones4[3] , 0 ) # midtones b
		]

		highlights3 = [
			max( highlights4[0] + highlights4[3] , 0 ), # highlights r
			max( highlights4[1] + highlights4[3] , 0 ), # highlights g
			max( highlights4[2] + highlights4[3] , 0 ) # highlights b
		]

		self.shadows = shadows3
		self.midtones = midtones3
		self.highlights = highlights3
		return

	def set_slopeOffsetPower(self, slope, offset, power):
		'''
		slope: float
		offset: float
		power: float
		'''
		minval = 1e-5
		self.slope = max(minval, slope)
		self.offset = offset
		self.power = max(minval, power)
		return

	def set_contrast(self, contrast):
		'''
		contrast: float
		'''
		self.contrast = math.clamp(contrast, -2.0, 2.0)
		return

	def set_vibrance(self, vibrance):
		'''
		vibrance: float
		'''
		self.vibrance = math.clamp(vibrance, -0.0, 2.0)
		return
	
	def set_saturation(self, saturation):
		'''
		saturation: float
		'''
		self.saturation = math.clamp(saturation, 0.0, 2.0)
		return

	def set_curves(self, shadowGamma, midPoint, highlightScale):
		'''
		shadowGamma: float3, midPoint: float3, highlightScale: float3
		'''
		minval = 1e-5
		self.shadowGamma = max(minval, shadowGamma)
		self.midPoint = max(minval, midPoint)
		self.highlightScale = highlightScale
		return

	def set_outputColorSpace(self, outputColorSpace):
		'''
		outputColorSpace: 'Rec709-sRGB-D65' is the only option for now
		TODO: add the other output color spaces.
		'''
		assert outputColorSpace in ['Rec709-sRGB-D65'] , "outputColorSpace must be 'Rec709-sRGB-D65'"
		self.outputColorSpace = outputColorSpace
		return


	def Set_quality(self, quality):
		'''
		quality: 'LOW', 'MEDIUM', 'HIGH', 'ULTRA'
		'''
		assert quality in ['LOW', 'MEDIUM', 'HIGH', 'ULTRA'] , "quality must be 'LOW', 'MEDIUM', 'HIGH', or 'ULTRA'"
		if quality == 'LOW':
			self.Set_dimension(16)
			self.Set_format('INTEGER')
		elif quality == 'MEDIUM':
			self.Set_dimension(32)
			self.Set_format('INTEGER')
		elif quality == 'HIGH':
			self.Set_dimension(32)
			self.Set_format('FLOAT')
		elif quality == 'ULTRA':
			self.Set_dimension(64)
			self.Set_format('FLOAT')
		
		return
	
	def Generate_LUT(self):
		results = []
		for b in range(self.dimension):
			for g in range(self.dimension):
				for r in range(self.dimension):
					
					# float3 v = float3{r, g, b} * (1.0f / float(config.lutDimension - 1u));
					v = np.array( [r, g, b] ) * (1.0 / (self.dimension - 1) )

					# // LogC encoding
					# v = LogC_to_linear(v);
					v = self.LogC_to_linear(v)

					# // Kill negative values near 0.0f due to imprecision in the log conversion 
					v = np.maximum(v, 0.0)

					if self.hasAdjustments:
						v = self.adjust_exposure(v, self.exposure)

						# // Purkinje shift ("low-light" vision)
						v = self.scotopicAdaptation(v, self.nightAdaptation)

					# // Move to color grading color space
					v = self.mat3x3_mul_vec3(self.selectColorGradingTransformIn, v)

					if self.hasAdjustments:

						# v = chromaticAdaptation(v, config.adaptationTransform);
						v = self.chromaticAdaptation(v, self.adaptationTransform(self.whiteBalance))

						# // Kill negative values before the next transforms
						v = np.maximum(v, 0.0)

						# // Channel mixer
						# v = channelMixer(v, builder->outRed, builder->outGreen, builder->outBlue);

						# // Shadows/mid-tones/highlights
						# v = tonalRanges(v, c.colorGradingLuminance,
						# 		builder->shadows, builder->midtones, builder->highlights,
						# 		builder->tonalRanges);

						# // The adjustments below behave better in log space
						v = self.linear_to_LogC(v)

                        # // ASC CDL
                        # v = colorDecisionList(v, builder->slope, builder->offset, builder->power);

                        # // Contrast in log space
                        # v = contrast(v, builder->contrast);

						# // Back to linear space
						v = self.LogC_to_linear(v)

						# // Vibrance in linear space
						# v = vibrance(v, c.colorGradingLuminance, builder->vibrance);

						# // Saturation in linear space
						# v = saturation(v, c.colorGradingLuminance, builder->saturation);

						# // Kill negative values before curves
						v = np.maximum(v, 0.0)

					# // RGB curves
					# v = curves(v,builder->shadowGamma, builder->midPoint, builder->highlightScale);
		

					if self.luminanceScaling == True:
						v = self.luminanceScaling_(v, self.toneMapper, self.selectColorGradingLuminance)
					else:
						v = self.toneMapper(v, 1)


					# // Go back to display color space
					# v = c.colorGradingOut * v;
					v = self.mat3x3_mul_vec3(self.selectColorGradingTransformOut, v)

					# // Apply gamut mapping
					if (self.gamutMapping == True):
						# // TODO: This should depend on the output color space
						v = self.gamut_mapping_sRGB(v)

					# // We need to clamp for the output transfer function
					# v = saturate(v);
					v = self.saturate_n(v)

					# // Apply OETF
					# v = c.oetf(v);
					v = self.selectOETF(v)

					results += list(v)
		
		# Convert the list to a NumPy array
		image = np.array(results)
		image = image.astype(np.float32)

		# Reshape the array into a 32x32x32x3 LUT cube
		# lut_cube = data_array.reshape(32, 32, 32, 3)	

		# Rearrange the LUT cube into a 1024x32 RGB image
		# image = np.vstack([lut_cube[:, :, i, :] for i in range(32)])

		image = image.reshape(self.dimension, 1024, 3)

		

		# Check the shape of the image
		# print(image.shape)  # This should output (1024, 32, 3)
		# print(image.dtype)

		return image




class MaterialParams:

	def __init__(self):

		self._max_kernel_size = 32
		
		self.baseColor = [0.0, 0.0, 0.0]
		self.alpha = 0.0

		self.metallic = 0.0
		self.roughness = 0.0
		self.reflectance = 0.0

		self.sheenColor = [0.0, 0.0, 0.0]
		self.sheenRoughness = 0.0

		self.clearcoat = 0.0
		self.clearcoatRoughness = 0.0
		self.clearCoatNormalStrength = 0.0

		self.anisotropy = 0.0
		self.anisotropyDirection = [0.0, 0.0, 0.0]

		self.ambientOcclusion = 0.0
		self.normalStrength = 0.0

		self.emissive = [0.0, 0.0, 0.0, 0.0]
		self.emissiveLuminance = 0.0 # this is not a filament parameter, but we use it to scale the emissive color from texture
		self.postLightingColor = [0.0, 0.0, 0.0, 0.0]

		self.ior = 0.0
		self.transmission = 0.0
		self.absorption = [0.0, 0.0, 0.0]
		self.thickness = 0.0
		self.subsurfacePower = 0.0
		self.subsurfaceColor = [0.0, 0.0, 0.0]

		self.maskThreshold = 0.0
		self.specularAntiAliasingVariance = 0.0
		self.specularAntiAliasingThreshold = 0.0
		self.doubleSided = 0

		self.axis = [0.0, 0.0]
		self.level = 0.0
		self.layer = 0.0
		self.reinhard = 0.0
		self.count = 0
		self.kernel = [[0.0, 0.0] for _ in range(self._max_kernel_size)]

		self.constantColor = 0
		self.showSun = 0
		self.color = [0.0, 0.0, 0.0, 1.0] # for environment lighting uniform color.

	
	def sRGB_to_linear(self, c):
		# assumes c is in the range [0, 1]
		if c <= 0.04045:
			c = c / 12.92
		else:
			c = pow((c + 0.055) / 1.055, 2.4)
		return c
	
	def absorption_at_distance(self, c, d):
		# c = color channel 0-1, d = distance.
		c = max(c, 1e-5)
		return -math.log(c) / max(1e-5, d)
	
	def compute_gaussian_coefficients(self, kernel_width: int, sigma: float, kernel_storage_size: int) -> Tuple[List[Tuple[float, float]], int]:
		alpha = 1.0 / (2.0 * sigma * sigma)

		# number of positive-side samples needed, using linear sampling
		m = (kernel_width - 1) // 4 + 1
		# clamp to what we have
		m = min(kernel_storage_size, m)

		# How the kernel samples are stored:
		#  *===*---+---+---+---+---+---+
		#  | 0 | 1 | 2 | 3 | 4 | 5 | 6 |       Gaussian coefficients (right size)
		#  *===*-------+-------+-------+
		#  | 0 |   1   |   2   |   3   |       stored coefficients (right side)

		kernel = [(0.0, 0.0)] * kernel_storage_size
		kernel[0] = (1.0, 0.0)
		total_weight = kernel[0][0]

		for i in range(1, m):
			x0 = i * 2 - 1
			x1 = i * 2
			k0 = math.exp(-alpha * x0 * x0)
			k1 = math.exp(-alpha * x1 * x1)

			k = k0 + k1
			o = k1 / k
			kernel[i] = (k, o)
			total_weight += (k0 + k1) * 2.0

		for i in range(m):
			kernel[i] = (kernel[i][0] / total_weight, kernel[i][1])

		return kernel, m

	def luminance_from_ev(self, ev100):
		# With L the average scene luminance, S the sensitivity and K the
		# reflected-light meter calibration constant:
		#
		# EV = log2(L * S / K)
		# L = 2^EV100 * K / 100
		#
		# As in ev100FromLuminance(luminance), we use K = 12.5 to match common camera
		# manufacturers (Canon, Nikon and Sekonic):
		#
		# L = 2^EV100 * 12.5 / 100 = 2^EV100 * 0.125
		#
		# With log2(0.125) = -3 we have:
		#
		# L = 2^(EV100 - 3)
		#
		# Reference: https://en.wikipedia.org/wiki/Exposure_value
		return math.pow(2.0, ev100 - 3.0)


	def Set_baseColor(self, r, g, b):
		self.baseColor = [self.sRGB_to_linear(r), self.sRGB_to_linear(g), self.sRGB_to_linear(b)]

	def Set_alpha(self, a):
		self.alpha = a
	
	def Set_metallic(self, m):
		self.metallic = m
	
	def Set_roughness(self, r):
		self.roughness = r
	
	def Set_reflectance(self, r):
		self.reflectance = r
	
	def Set_sheenColor(self, r, g, b):
		self.sheenColor = [self.sRGB_to_linear(r), self.sRGB_to_linear(g), self.sRGB_to_linear(b)]
	
	def Set_sheenRoughness(self, r):
		self.sheenRoughness = r
	
	def Set_clearcoat(self, c):
		self.clearcoat = c
	
	def Set_clearcoatRoughness(self, r):
		self.clearcoatRoughness = r
	
	def Set_clearCoatNormalStrength(self, s):
		self.clearCoatNormalStrength = s

	def Set_anisotropy(self, a):
		self.anisotropy = a
	
	def Set_anisotropyDirection(self, x, y, z):
		self.anisotropyDirection = [x, y, z]
	
	def Set_ambientOcclusion(self, a):
		self.ambientOcclusion = a
	
	def Set_normalStrength(self, s):
		self.normalStrength = s
	
	def Set_emissive(self, r, g, b, ev, exposureweight):
		luminance = self.luminance_from_ev( ev )
		self.emissiveLuminance = luminance
		self.emissive = [
			self.sRGB_to_linear(r) * luminance, 
			self.sRGB_to_linear(g) * luminance, 
			self.sRGB_to_linear(b) * luminance, 
			exposureweight]
	
	def Set_postLightingColor(self, r, g, b, a):
		self.postLightingColor = [self.sRGB_to_linear(r), self.sRGB_to_linear(g), self.sRGB_to_linear(b), a]
	
	def Set_ior(self, i):
		self.ior = i
	
	def Set_transmission(self, t):
		self.transmission = t
	
	def Set_absorption(self, r, g, b, d):
		r = self.absorption_at_distance( self.sRGB_to_linear(r) , d )
		g = self.absorption_at_distance( self.sRGB_to_linear(g) , d )
		b = self.absorption_at_distance( self.sRGB_to_linear(b) , d )
		self.absorption = [r, g, b]
	
	def Set_thickness(self, t):
		self.thickness = t
	
	def Set_subsurfacePower(self, p):
		self.subsurfacePower = p
	
	def Set_subsurfaceColor(self, r, g, b):
		self.subsurfaceColor = [self.sRGB_to_linear(r), self.sRGB_to_linear(g), self.sRGB_to_linear(b)]
	
	def Set_maskThreshold(self, t):
		self.maskThreshold = t
	
	def Set_specularAntiAliasingVariance(self, v):
		self.specularAntiAliasingVariance = v
	
	def Set_specularAntiAliasingThreshold(self, t):
		self.specularAntiAliasingThreshold = t
	
	def Set_doubleSided(self, s):
		self.doubleSided = s
	
	def Set_axis(self, width, height, level, axis):
		'''
		width = width of the input texture
		height = height of the input texture
		level = desired mip level to generate
		axis = 0 for x, 1 for y
		'''
		level += 1 # make it 1 based.
		x_mask = axis == 0
		y_mask = axis == 1
		width2 = width  * x_mask * ( 1 / level )
		height2 = height * y_mask * ( 1 / level )
		width_reciprocal  = (1.0 / width2) if width2 >  0 else 0.0
		height_reciprocal = (1.0 / height2) if height2 > 0 else 0.0
		self.axis = [width_reciprocal, height_reciprocal]
	
	def Set_level(self, l):
		self.level = l

	def Set_layer(self, l):
		self.layer = l

	def Set_reinhard(self, r):
		self.reinhard = r
	
	def Set_count(self, c):
		# not the intended way to set the count, but it's here for completeness
		self.count = c
	
	def Set_kernel(self, k):
		# not the intended way to set the kernel, but it's here for completeness
		self.kernel = k
	
	def Set_GaussianKernel(self):
		kernel_size = 23
		sigma = 3.6666666666666666666666666666667 # no idea why but this magic number makes the resulting kernel match the one from renderdoc.
		max_size = self._max_kernel_size
		result = self.compute_gaussian_coefficients(kernel_size, sigma, max_size)
		self.count = result[1]
		kernel = [ each for each in result[0] ]
		# flatten kernel array and assign to self.kernel
		self.kernel = []
		for each in kernel:
			self.kernel.append([each[0],each[1]])
	
	def Set_constantColor(self, state):
		self.constantColor = int(state)
	
	def Set_showSun(self, state):
		self.showSun = int(state)
	
	def Set_color(self, r,g,b,a):
		self.color = [self.sRGB_to_linear(r), self.sRGB_to_linear(g), self.sRGB_to_linear(b), a]
	
	def Print_All(self):
		
		print("baseColor: " + str(self.baseColor))
		print("alpha: " + str(self.alpha))
		print("metallic: " + str(self.metallic))
		print("roughness: " + str(self.roughness))
		print("reflectance: " + str(self.reflectance))
		print("sheenColor: " + str(self.sheenColor))
		print("sheenRoughness: " + str(self.sheenRoughness))
		print("clearcoat: " + str(self.clearcoat))
		print("clearcoatRoughness: " + str(self.clearcoatRoughness))
		print("clearCoatNormalStrength: " + str(self.clearCoatNormalStrength))
		print("anisotropy: " + str(self.anisotropy))
		print("anisotropyDirection: " + str(self.anisotropyDirection))
		print("ambientOcclusion: " + str(self.ambientOcclusion))
		print("normalStrength: " + str(self.normalStrength))
		print("emissive: " + str(self.emissive))
		print("emissiveLuminance: " + str(self.emissiveLuminance))
		print("postLightingColor: " + str(self.postLightingColor))
		print("ior: " + str(self.ior))
		print("transmission: " + str(self.transmission))
		print("absorption: " + str(self.absorption))
		print("thickness: " + str(self.thickness))
		print("subsurfacePower: " + str(self.subsurfacePower))
		print("subsurfaceColor: " + str(self.subsurfaceColor))
		print("maskThreshold: " + str(self.maskThreshold))
		print("specularAntiAliasingVariance: " + str(self.specularAntiAliasingVariance))
		print("specularAntiAliasingThreshold: " + str(self.specularAntiAliasingThreshold))
		print("doubleSided: " + str(self.doubleSided))
		print("axis: " + str(self.axis))
		print("level: " + str(self.level))
		print("layer: " + str(self.layer))
		print("reinhard: " + str(self.reinhard))
		print("count: " + str(self.count))
		print("kernel: " + str(self.kernel))

		return
	
	def To_List(self, buffer_len = 256):
		
		kernel_start = buffer_len // 2

		ret = []

		ret += self.baseColor # 0, 1, 2
		ret += [self.metallic] # 3
		ret += [self.roughness] # 4
		ret += [self.reflectance] # 5
		ret += self.sheenColor # 6, 7, 8
		ret += [self.sheenRoughness] # 9
		ret += [self.clearcoat] # 10
		ret += [self.clearcoatRoughness] # 11
		ret += [self.clearCoatNormalStrength] # 12
		ret += [self.anisotropy] # 13
		ret += self.anisotropyDirection # 14, 15, 16
		ret += [self.ambientOcclusion] # 17
		ret += [self.normalStrength] # 18
		ret += self.emissive # 19, 20, 21, 22
		ret += self.postLightingColor # 23, 24, 25, 26
		ret += [self.ior] # 27
		ret += [self.transmission] # 28
		ret += self.absorption # 29, 30, 31
		ret += [self.thickness] # 32
		ret += [self.subsurfacePower] # 33
		ret += self.subsurfaceColor # 34, 35, 36
		ret += [self.maskThreshold] # 37
		ret += [self.specularAntiAliasingVariance] # 38
		ret += [self.specularAntiAliasingThreshold] # 39
		ret += [self.doubleSided] # 40
		ret += self.axis # 41, 42
		ret += [self.level] # 43
		ret += [self.layer] # 44
		ret += [self.reinhard] # 45
		ret += [self.count] # 46
		ret += [self.emissiveLuminance] # 47
		ret += [self.alpha] # 48
		ret += [self.constantColor] # 49
		ret += [self.showSun] # 50
		ret += self.color # 51, 52, 53, 54

		# self.Print_All()

		# right pad the ret list with zeros to get the length to 128.
		ret += [0 for _ in range(kernel_start-len(ret))]
		
		# kernel is a list of 32 vec2,s so we need to flatten it.. 
		kernel_flattened = [item for sublist in self.kernel for item in sublist]
		ret += kernel_flattened

		# right pad the ret list with zeros to get the length to 256.
		ret += [0 for _ in range(buffer_len-len(ret))]

		return ret
	

class Templateext:
	"""
	Templateext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

	def MaterialParamsTemplate(self):
		# return an instance of the MaterialParams class.
		return MaterialParams()

	def ColorgradingParamsTemplate(self):
		# return an instance of the MaterialParams class.
		return ColorgradingParams()