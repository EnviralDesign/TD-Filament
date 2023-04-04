import td
import inspect # this might not be neccesary if running inside of TD.

class EditorExt:
	"""
	EditorExt description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.outliner = self.ownerComp.op('outliner')
		self.outlinerWrapper = self.ownerComp.op('container_outliner_wrapper')
		self.attrEditor = self.ownerComp.op('container_UG_V4')

		# initialize container to sensible default size.
		self.ownerComp.par.w = 1280
		self.ownerComp.par.h = 720
		

	@property
	def OutlinerWrapper(self):
		return self.outlinerWrapper

	@property
	def Outliner(self):
		return self.outliner

	@property
	def AttributeEditor(self):
		return self.attrEditor