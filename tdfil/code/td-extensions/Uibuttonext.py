

class UiButton:
	"""
	UiButton description
	"""
	
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp

		self.Value0_PAR = self.ownerComp.par.Value0
		self.Callback_DAT = self.ownerComp.par.Callbackdat.eval()

		self.delayed_script_DAT = self.ownerComp.op('delayed_script')
		self.Buttonmode_PAR = self.ownerComp.par.Buttonmode
		self.Unclickframedelay_PAR = self.ownerComp.par.Unclickframedelay

	def Update_Callback_Dat(self):
		self.Callback_DAT = self.ownerComp.par.Callbackdat.eval()
		return

	def Reset(self):

		Buttonmode = self.Buttonmode_PAR.eval()

		if Buttonmode in ['momentary','toggle']:
			self.Value0_PAR.val = self.Value0_PAR.default
		elif Buttonmode == 'radio':
			debug('radio buttons Reset() not implemented yet..')

	def Click(self):
		
		Buttonmode = self.Buttonmode_PAR.eval()
		Unclickframedelay = self.Unclickframedelay_PAR.eval()

		if Buttonmode == 'momentary':
			self.Value0_PAR.val = True
			self.delayed_script_DAT.text = 'parent.ui.par.Value0 = 0'
			self.delayed_script_DAT.run(delayFrames=Unclickframedelay)

		elif Buttonmode == 'toggle':
			self.Value0_PAR.val = not self.Value0_PAR.eval()

		elif Buttonmode == 'radio':
			debug('radio buttons Click() not implemented yet..')



	def Set(self, val):
		
		Buttonmode = self.Buttonmode_PAR.eval()

		if Buttonmode in ['momentary','toggle']:
			self.Value0_PAR.val = val
		
		elif Buttonmode in ['radio']:
			debug('radio buttons Set() not implemented yet..')



	def Callback(self, val):
		
		if self.Callback_DAT != None:
			self.Callback_DAT.run(val)


