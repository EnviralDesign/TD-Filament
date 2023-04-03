

class envext:
	"""
	envext description
	"""
	def __init__(self, ownerComp):
		# The component to which this extension is attached
		self.ownerComp = ownerComp
		self.folderDAT = op('folder_get_assets')


	def Asset_Health(self, verbosity):
		'''
		This function returns overall health / stats about an asset. Primarily used to 
		determine if there are missing or incorrect inputs, errors, etc that the user should attend to.

		verbosity = 0 means you get the MAX() error. so if there are 2 greens, and 1 yellow, you'll get yellow back.
		verbosity = 1 means you will get a dictionary of information back, with a green/yellow/red status for each.
		'''
		basenames = list(map(str,self.folderDAT.col('basename')[1::]))

		# determine highest mip level we need to check. this can vary depending on the quality level it was exported with
		mips = [x for x in basenames if x.startswith('m') and x[1].isdigit() ]
		mips = sorted(list(map(int,set([ x[1] for x in mips ]))))
		highestMip = max(mips)

		status = {}

		# check for dfg and sh files.
		status['dfg_lookup_texture_check'] = [2,0]['dfg' in basenames]
		status['sh_file_check'] = [2,0]['sh' in basenames]

		# there are 6 faces for every mip level of the cube map. iterate through them all and check if they exist.
		# NOTE: we are determining what the highest mip is above, by looking for the highest integer in any file name.
		for mip in range(highestMip):
			expectedMips = [ 	
				f'm{mip}_px', f'm{mip}_nx',
				f'm{mip}_py', f'm{mip}_ny',
				f'm{mip}_pz', f'm{mip}_nz']
			for expectedMip in expectedMips:
				status[expectedMip+'_file_check'] = [2,0][expectedMip in basenames]
		
		# check the 6 faces of the skybox/environment cube map.
		status['px'] = [2,0]['px' in basenames]
		status['nx'] = [2,0]['nx' in basenames]
		status['py'] = [2,0]['py' in basenames]
		status['ny'] = [2,0]['ny' in basenames]
		status['pz'] = [2,0]['pz' in basenames]
		status['nz'] = [2,0]['nz' in basenames]

		if verbosity == 0:
			return max( status.values() )

		else:
			return status