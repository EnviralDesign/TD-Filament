# me - this DAT
# 
# comp - the List Component that holds this panel
# row - the row number of the cell being updated
# col - the column number of the cell being updated
#
# attribs contains the following members:
#
# text				   str            cell contents
# help                 str       	  help text
#
# textColor            r g b a        font color
# textOffsetX		   n			  horizontal text offset
# textOffsetY		   n			  vertical text offset
# textJustify		   m			  m is one of:  JustifyType.TOPLEFT, JustifyType.TOPCENTER,
#													JustifyType.TOPRIGHT, JustifyType.CENTERLEFT,
#													JustifyType.CENTER, JustifyType.CENTERRIGHT,
#													JustifyType.BOTTOMLEFT, JustifyType.BOTTOMCENTER,
#													JustifyType.BOTTOMRIGHT
#
# bgColor              r g b a        background color
#
# leftBorderInColor	   r g b a		  inside left border color
# rightBorderInColor   r g b a		  inside right border color
# bottomBorderInColor  r g b a		  inside bottom border color
# topBorderInColor	   r g b a		  inside top border color
#
# leftBorderOutColor   r g b a		  outside left border color
# rightBorderOutColor  r g b a		  outside right border color
# bottomBorderOutColor r g b a		  outside bottom border color
# topBorderOutColor	   r g b a		  outside top border color
#
# colWidth             w              sets column width
# colStretch            True/False     sets column stretchiness (width is min width)
# rowHeight            h              sets row height
# rowStretch            True/False     sets row stretchiness (height is min height)
# rowIndent            w              offsets entire row by this amount
#
# editable			   int			  number of clicks to activate editing the cell.
# draggable             True/False     allows cell to be drag/dropped elsewhere
# fontBold             True/False     render font bolded
# fontItalic           True/False     render font italicized
# fontSizeX            float		  font X size in pixels
# fontSizeY            float		  font Y size in pixels, if not specified, uses X size
# sizeInPoints         True/False	  If true specify font size in points, rather than pixels.
# fontFace             str			  font face, example 'Verdana'
# fontFile             str			  font file, when specified on disk or embedded in VFS.
# wordWrap             True/False     word wrap
#
# top                  TOP			  background TOP operator
#
# select   true when the cell/row/col is currently being selected by the mouse
# rollover true when the mouse is currently over the cell/row/col
# radio    true when the cell/row/col was last selected
# focus    true when the cell/row/col is being edited
#
#

# called when Reset parameter is pulsed, or on load

sceneListDat = op('null_scene_list')

indentation_col = 0
expand_col = 1
icon_col = 2
name_col = 3

def onInitCell(comp, row, col, attribs):

	# grab the cell content from the Folder DAT
	depth_current = int(sceneListDat[row+1,'Depth'])
	depth_next = int(sceneListDat[row+2,'Depth'] if sceneListDat[row+2,'Depth'] != None else -1)

	# indentation column
	if col == indentation_col:
		if depth_current > 0:
			if depth_next >= 0:
				if depth_next < depth_current and depth_current > 1:
					attribs.top = op(f'null_parallel_wire_{depth_current}')
					# attribs.bgColor = [0.5,0.5,0,.2] # YELLOW
					pass
				else:
					attribs.top = op(f'null_parallel_wire_{depth_current}')
					# attribs.bgColor = [0,0.5,0.5,.2] # TEAL
					pass
			else:
				# attribs.top = op(f'null_last_child_{depth_current}')
				# attribs.bgColor = [0.8,0.2,0.5,.2] # PINK
				pass
			pass

	# expand/collapse column
	if col == expand_col:
		numChildren =  int(sceneListDat[row+1,'Numchildren'])
		Numinstances =  int(sceneListDat[row+1,'Numinstances'])
		opId =  int(sceneListDat[row+1,'id'])
		if max(numChildren,Numinstances) > 0:
			if opId not in parent.outliner.Collapsed:
				attribs.top = op('collapse')
			else:
				attribs.top = op('expand')
				pass
			# attribs.bgColor = [0,0,1,.2] # BLUE
			pass
		else:
			if depth_current == 0:
				attribs.top = op(f'null_intermediate_child_{depth_current}')
				# attribs.bgColor = [1,0.5,0,.2] # ORANGE
			else:
				if depth_next == depth_current:
					attribs.top = op(f'null_intermediate_child_{0}')
					# attribs.bgColor = [1,0,0,.2] # RED
					pass
				elif depth_next < depth_current:
					attribs.top = op(f'null_last_child_{0}')
					# attribs.bgColor = [0.5,0,0.5,.2] # PURPLE
					pass
			pass

	# icon column
	elif col == icon_col:
		attribs.top = op(sceneListDat[row+1,'thumbnail'])

	# primary name column
	elif col == name_col:
		attribs.text = cellContent = sceneListDat[row+1,'name'].val
		attribs.editable = 2

	return

def onInitRow(comp, row, attribs):
	# if this is the first row make the background slightly red, otherwise keep it grey
	attribs.rowHeight = op.TDFIL.Style('rowheight',0) + op.TDFIL.Style('rowspacing',0)
	attribs.textOffsetX = op.TDFIL.Style('textpadding',0)

	numIndents = int(sceneListDat[row+1,'Depth'])
	attribs.rowIndent = op.TDFIL.Style('indentsize',0) * numIndents

	sceneListCell = sceneListDat[row+1,'selected']
	if sceneListCell != None:
		row_interact_state = int(sceneListDat[row+1,'selected']) + ((parent.outliner.rolloverRow == row) * parent.outliner.panel.inside)
	else:
		row_interact_state = 0

	style_r = 'currentcolorr' if int(sceneListDat[row+1,'current']) == 1 else 'bgcolorr'
	style_g = 'currentcolorg' if int(sceneListDat[row+1,'current']) == 1 else 'bgcolorg'
	style_b = 'currentcolorb' if int(sceneListDat[row+1,'current']) == 1 else 'bgcolorb'
	style_a = 'currentalpha' if int(sceneListDat[row+1,'current']) == 1 else 'bgalpha'

	# assign the bgColor to the rows attributes
	attribs.bgColor = [
		op.TDFIL.Style(style_r,row_interact_state),
		op.TDFIL.Style(style_g,row_interact_state),
		op.TDFIL.Style(style_b,row_interact_state),
		op.TDFIL.Style(style_a,row_interact_state)]

	attribs.bottomBorderOutColor = [
		op.TDFIL.Style('platecolorr',row_interact_state),
		op.TDFIL.Style('platecolorg',row_interact_state),
		op.TDFIL.Style('platecolorb',row_interact_state),
		op.TDFIL.Style('platealpha',row_interact_state)]

	attribs.textColor = [
		op.TDFIL.Style('fontcolorr',row_interact_state),
		op.TDFIL.Style('fontcolorg',row_interact_state),
		op.TDFIL.Style('fontcolorb',row_interact_state),
		op.TDFIL.Style('fontalpha',row_interact_state)]

	return

def onInitCol(comp, col, attribs):

	# expand/collapse column
	if col == expand_col:
		attribs.colStretch = False
		attribs.colWidth = op.TDFIL.Style('indentsize',0)

	# indentation column
	if col == indentation_col:
		attribs.colStretch = False
		attribs.colWidth = op.TDFIL.Style('indentsize',0)

	# icon column
	elif col == icon_col:
		attribs.colStretch = False
		attribs.colWidth = op.TDFIL.Style('indentsize',0) + op.TDFIL.Style('rowspacing',0)

	# primary name column
	elif col == name_col:
		attribs.colStretch = True
		attribs.colWidth = 100 # not used if colstretch is True.
		attribs.textJustify = JustifyType.CENTERLEFT

	return

def onInitTable(comp, attribs):

	attribs.sizeInPoints = True
	attribs.fontSizeX = op.TDFIL.Style('fontpointsize',0)
	attribs.fontFace = str(op.TDFIL.Style('fontminor',0))
	return

# called during specific events
#
# coords - a named tuple containing the following members:
#   x
#   y
#   u
#   v

def onRollover(comp, row, col, coords, prevRow, prevCol, prevCoords):
	
	# current
	row_interact_state = 2 if int(sceneListDat[row+1,'selected']) == 1 else 1
	style_r = 'currentcolorr' if int(sceneListDat[row+1,'current']) == 1 else 'bgcolorr'
	style_g = 'currentcolorg' if int(sceneListDat[row+1,'current']) == 1 else 'bgcolorg'
	style_b = 'currentcolorb' if int(sceneListDat[row+1,'current']) == 1 else 'bgcolorb'
	style_a = 'currentalpha' if int(sceneListDat[row+1,'current']) == 1 else 'bgalpha'
	parent.outliner.rowAttribs[row].bgColor = [
		op.TDFIL.Style(style_r,row_interact_state),
		op.TDFIL.Style(style_g,row_interact_state),
		op.TDFIL.Style(style_b,row_interact_state),
		op.TDFIL.Style(style_a,row_interact_state)]
	
	# try excepts prevent errors when items are deleted and user is interacting with outliner at same time.
	try:
		prev_selected_state = int(sceneListDat[prevRow+1,'selected'])	
	except:
		prev_selected_state = 0
	
	try:
		prev_current_state = int(sceneListDat[prevRow+1,'current'])	
	except:
		prev_current_state = 0
	
	if prev_current_state > 0:
		style_r = 'currentcolorr'
		style_g = 'currentcolorg'
		style_b = 'currentcolorb'
		style_a = 'currentalpha'
	
	else:
		style_r = 'bgcolorr'
		style_g = 'bgcolorg'
		style_b = 'bgcolorb'
		style_a = 'bgalpha'

	row_interact_state = max( prev_selected_state , prev_current_state )

	if row != prevRow:
		if parent.outliner.rowAttribs[prevRow] != None:
			parent.outliner.rowAttribs[prevRow].bgColor = [
				op.TDFIL.Style(style_r,row_interact_state),
				op.TDFIL.Style(style_g,row_interact_state),
				op.TDFIL.Style(style_b,row_interact_state),
				op.TDFIL.Style(style_a,row_interact_state)]

	return

def onSelect(comp, startRow, startCol, startCoords, endRow, endCol, endCoords, start, end):
	
	# this happens so that if the user clicks on an empty part of the outliner, on release we can know if we should deselect all, or not.
	parent.outliner.store('row_clicked', True)

	modeLookup = {
		3:'selection',
		1:'expandcollapse',
		}

	# selection began
	if start == True:
		me.store( 'shift', bool(parent.outliner.panel.shift) )
		me.store( 'ctrl', bool(parent.outliner.panel.ctrl) )
		me.store( 'mode', modeLookup.get(startCol,None) )
		me.store( 'Objtype', int(sceneListDat[endRow+1,'Objtype']) )

	# selection end
	elif end == True:

		# use the modifier states as they were when the selection began.
		shift = me.fetch('shift', False)
		ctrl = me.fetch('ctrl', False)
		mode = me.fetch('mode', None)
		Objtype = me.fetch('Objtype', -1)

		if mode == 'selection':
			
			# STICKY CAMERA
			if startRow == endRow and Objtype in op.TDFIL.Type_Group('CAMERA'):
				parent.Editor.Scene.Deselect_All_Objects()
				parent.Editor.Scene.Deselect_All_Instances()
				parent.Editor.Camera.selected = True
				
			# STICKY LIGHT MANAGER
			elif startRow == endRow and Objtype in op.TDFIL.Type_Group('LIGHTMANAGER'):
				parent.Editor.Scene.Deselect_All_Objects()
				parent.Editor.Scene.Deselect_All_Instances()
				parent.Editor.SettingsManager.selected = True
			
			# the general case. this is 99% of the time.
			else:

				sel = []
				
				if endRow >= startRow:
					a,b = startRow, endRow+1
					sel_reversed = False
				else:
					b,a = startRow+1, endRow
					sel_reversed = True
				for i in range(a,b):
					sel += [ op(sceneListDat[i+1,'path']) ]
				
				if sel_reversed == True:
					sel = sel[::-1]

				# if no modifiers are down
				if max(shift,ctrl) == False:
					parent.Editor.Scene.Select_Objects(sel,clear_previous=True)

				# if shift only is down
				elif shift == True and ctrl == False:
					parent.Editor.Scene.Select_Objects(sel,clear_previous=False)

				# if ctrl only is down
				elif shift == False and ctrl == True:
					parent.Editor.Scene.Deselect_Objects( sel )

		elif mode == 'expandcollapse':
			
			topref = parent.outliner.cellAttribs[startRow,startCol].top
			if topref != None:
				if topref.name in ['expand','collapse']:
					opref = op(sceneListDat[startRow+1,'path'])
					parent.outliner.ExpandCollapseToggle( [opref.id] )
			pass

	return

def onRadio(comp, row, col, prevRow, prevCol):
	return

def onFocus(comp, row, col, prevRow, prevCol):
	return

def onEdit(comp, row, col, val):
	# print(row,col,val)
	targetOP = op(sceneListDat[row+1,'path'])
	Objtype = int(sceneListDat[row+1,'Objtype'])

	if Objtype in op.TDFIL.Type_Group('SYSTEM'):
		debug('cannot rename system nodes...')
		return

	if targetOP != None:
		targetOP.name = tdu.legalName( val )
	else:
		debug(f'could not rename op {str(sceneListDat[row+1,"path"])} because op was not valid...')
	return

# return True if interested in this drop event, False otherwise
def onHoverGetAccept(comp, row, col, coords, prevRow, prevCol, prevCoords, dragItems):
	return False
def onDropGetAccept(comp, row, col, coords, prevRow, prevCol, prevCoords, dragItems):
	return False

	