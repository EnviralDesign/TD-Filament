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
		
		self.uitemplate_BASE = self.ownerComp.op('Ui_Templates')

		self.style = {
			'bgcolorr':[0.1,0.2,0.4],
			'bgcolorg':[0.1,0.2,0.4],
			'bgcolorb':[0.1,0.2,0.4],
			'bgalpha':[1.0]*3,

			'currentcolorr':[0.1,0.3,0.4],
			'currentcolorg':[0.1,0.15,0.2],
			'currentcolorb':[0.1,0.1,0.1],
			'currentalpha':[1.0]*3,

			'platecolorr':[0.05]*3,
			'platecolorg':[0.05]*3,
			'platecolorb':[0.05]*3,
			'platealpha':[1.0]*3,

			'fontcolorr':[.9]*3,
			'fontcolorg':[.9]*3,
			'fontcolorb':[.9]*3,
			'fontalpha':[1.0]*3,

			'fontminor':['Tahoma'],

			'fontpointsize':[9],
			'textpadding':[4],
			'indentsize':[20],
			'rowheight':[25],
			'rowspacing':[1],
		}
	
	@property
	def Template_UiHeader(self):
		return self.uitemplate_BASE.op('ui_header')

	@property
	def Template_UiButton(self):
		return self.uitemplate_BASE.op('ui_button')


	def Style(self, par, styleState):
		'''
		par is a ref to the par. can grab this easily within a parameter expression by using me.curPar
		styleState is an integer defining what state the interaction is in. 0 is base, 1 is hover, 2 is click, etc.
		'''
		
		if isinstance(par, td.Par): # if par is a td.Par, get it's name and use that.
			par_style = self.style.get(par.name,None)
		elif isinstance(par, str): # if par is a string, we can just use it directly as is.
			par_style = self.style.get(par,None)
		else:
			debug(f'{type(par)} type not recognized... skipping.')

		if par_style == None:
			debug(f'par:{par.name} does not exist in self.style dict, must add - returning zero.')
			return 0

		styleState = int(styleState)
		try:
			return par_style[styleState]
		except:
			debug(f'style_state <{styleState}> not present in self.style dictionary, returning zero.')
			return par_style[0]

		return 

	@property
	def OutlinerWrapper(self):
		return self.outlinerWrapper

	@property
	def Outliner(self):
		return self.outliner

	@property
	def AttributeEditor(self):
		return self.attrEditor

	def TraceFunctionCall(self):
		'''
		This function will print the entire function call's 
		stack trace showing what called what where.
		Obviously if you use scriptDat.run() 
		it will break the function call chain.
		'''
		inspectedStack = inspect.stack()
		print('qwe')
		source = []
		line = []
		function = []
		for f in inspectedStack[::-1]:
			frameInfo = inspect.getframeinfo(f[0])
			source += [frameInfo[0]]
			line += [str(frameInfo[1])]
			function += [frameInfo[2]]

		source_max = max( [len(x) for x in source] )
		line_max = max( [len(x) for x in line] )
		function_max = max( [len(x) for x in function] )

		print('-'*(source_max+line_max+function_max+11))

		for i,f in enumerate(zip(source,line,function)):
			f = list(f)
			source_ = f[0].ljust(source_max)
			line_ = f[1].ljust(line_max)
			function_ = (f[2]+'()').ljust(function_max)

			print('%i) %s : %s : %s'%(i,source_,line_,function_))

		print('-'*(source_max+line_max+function_max+11))

		return