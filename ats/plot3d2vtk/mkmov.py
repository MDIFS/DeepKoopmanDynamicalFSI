# trace generated using paraview version 5.8.1
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'PVD Reader'
u4pvd = PVDReader(FileName='/data/home/yamazaki/data5/DeepLearning/Pytorch_Training/Day9_trial02/ats/plot3d2vtk/u4.0/u4/u4.pvd')
u4pvd.PointArrays = ['Density', 'u', 'v', 'w', 'Pressure']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1397, 601]

# get layout
layout1 = GetLayout()

# show data in view
u4pvdDisplay = Show(u4pvd, renderView1, 'StructuredGridRepresentation')

# get color transfer function/color map for 'Density'
densityLUT = GetColorTransferFunction('Density')
densityLUT.RGBPoints = [0.8907005786895752, 0.231373, 0.298039, 0.752941, 0.9638705253601074, 0.865003, 0.865003, 0.865003, 1.0370404720306396, 0.705882, 0.0156863, 0.14902]
densityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Density'
densityPWF = GetOpacityTransferFunction('Density')
densityPWF.Points = [0.8907005786895752, 0.0, 0.5, 0.0, 1.0370404720306396, 1.0, 0.5, 0.0]
densityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
u4pvdDisplay.Representation = 'Surface'
u4pvdDisplay.ColorArrayName = ['POINTS', 'Density']
u4pvdDisplay.LookupTable = densityLUT
u4pvdDisplay.OSPRayScaleArray = 'Density'
u4pvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
u4pvdDisplay.SelectOrientationVectors = 'Density'
u4pvdDisplay.ScaleFactor = 1.6653757095336914
u4pvdDisplay.SelectScaleArray = 'Density'
u4pvdDisplay.GlyphType = 'Arrow'
u4pvdDisplay.GlyphTableIndexArray = 'Density'
u4pvdDisplay.GaussianRadius = 0.08326878547668458
u4pvdDisplay.SetScaleArray = ['POINTS', 'Density']
u4pvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
u4pvdDisplay.OpacityArray = ['POINTS', 'Density']
u4pvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
u4pvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
u4pvdDisplay.PolarAxes = 'PolarAxesRepresentation'
u4pvdDisplay.ScalarOpacityFunction = densityPWF
u4pvdDisplay.ScalarOpacityUnitDistance = 0.7755914827123932

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
u4pvdDisplay.ScaleTransferFunction.Points = [0.8907005786895752, 0.0, 0.5, 0.0, 1.0370404720306396, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
u4pvdDisplay.OpacityTransferFunction.Points = [0.8907005786895752, 0.0, 0.5, 0.0, 1.0370404720306396, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
u4pvdDisplay.DataAxesGrid.YTitle = 'Radius'
u4pvdDisplay.DataAxesGrid.XTitleFontFamily = 'Times'
u4pvdDisplay.DataAxesGrid.XTitleFontSize = 25
u4pvdDisplay.DataAxesGrid.YTitleFontFamily = 'Times'
u4pvdDisplay.DataAxesGrid.YTitleFontSize = 25
u4pvdDisplay.DataAxesGrid.ZTitleFontFamily = 'Times'
u4pvdDisplay.DataAxesGrid.ZTitleFontSize = 25
u4pvdDisplay.DataAxesGrid.GridColor = [0.0, 0.0, 0.0]
u4pvdDisplay.DataAxesGrid.XLabelFontFamily = 'Times'
u4pvdDisplay.DataAxesGrid.XLabelFontSize = 25
u4pvdDisplay.DataAxesGrid.YLabelFontFamily = 'Times'
u4pvdDisplay.DataAxesGrid.YLabelFontSize = 25
u4pvdDisplay.DataAxesGrid.ZLabelFontFamily = 'Times'
u4pvdDisplay.DataAxesGrid.ZLabelFontSize = 25

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0006275177001953125, 9999.902000002563, 0.0]
renderView1.CameraFocalPoint = [0.0006275177001953125, -0.09799999743700027, 0.0]
renderView1.CameraViewUp = [1.0, 0.0, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
u4pvdDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# set scalar coloring
ColorBy(u4pvdDisplay, ('POINTS', 'Pressure'))

# Hide the scalar bar for this color map if no visible data is colored by it.
HideScalarBarIfNotNeeded(densityLUT, renderView1)

# rescale color and/or opacity maps used to include current data range
u4pvdDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
u4pvdDisplay.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'Pressure'
pressureLUT = GetColorTransferFunction('Pressure')
pressureLUT.RGBPoints = [0.8894487619400024, 0.231373, 0.298039, 0.752941, 0.9696468710899353, 0.865003, 0.865003, 0.865003, 1.0498449802398682, 0.705882, 0.0156863, 0.14902]
pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Pressure'
pressurePWF = GetOpacityTransferFunction('Pressure')
pressurePWF.Points = [0.8894487619400024, 0.0, 0.5, 0.0, 1.0498449802398682, 1.0, 0.5, 0.0]
pressurePWF.ScalarRangeInitialized = 1

# reset view to fit data
renderView1.ResetCamera()

# Rescale transfer function
pressureLUT.RescaleTransferFunction(0.94, 1.02)

# Rescale transfer function
pressurePWF.RescaleTransferFunction(0.94, 1.02)

# get color legend/bar for pressureLUT in view renderView1
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView1)
pressureLUTColorBar.AutoOrient = 0
pressureLUTColorBar.Orientation = 'Horizontal'
pressureLUTColorBar.WindowLocation = 'AnyLocation'
pressureLUTColorBar.Title = 'Pressure'
pressureLUTColorBar.ComponentTitle = ''
pressureLUTColorBar.TitleFontSize = 25
pressureLUTColorBar.ScalarBarLength = 0.22707740884978156

# change scalar bar placement
pressureLUTColorBar.Position = [0.40753758052970657, 0.028319467554076536]
pressureLUTColorBar.ScalarBarLength = 0.2270774088497809

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0006275177001953125, -45.59605332176343, 0.0]
renderView1.CameraFocalPoint = [0.0006275177001953125, -0.09799999743700027, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraParallelScale = 2.5627490630279546

# save animation
SaveAnimation('/data/home/yamazaki/data5/DeepLearning/Pytorch_Training/Day9_trial02/ats/plot3d2vtk/u4.0/u4/Flow.png', renderView1, ImageResolution=[1396, 600],
    FrameWindow=[0, 197])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [0.0006275177001953125, -45.59605332176343, 0.0]
renderView1.CameraFocalPoint = [0.0006275177001953125, -0.09799999743700027, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraParallelScale = 2.5627490630279546

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
quit()
