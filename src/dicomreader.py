from __future__ import print_function
import os
import sys
import vtk
import pydicom


class DICOMImageReader(vtk.vtkImageReader2):

    def PrepareDICOMFrames(self, fileNames):
        frames = []
        for fileName in fileNames:
            try:
                frame = pydicom.read_file(fileName)
                location = None
                if hasattr(frame, 'SliceLocation'):
                    location = frame.SliceLocation
                else:
                    location = frame.ImagePositionPatient[2]

                frames.append({
                    'FileName': fileName,
                    'Dimensions': list(map(int, [frame.Columns, frame.Rows])),
                    'Location': float(location),
                    'Origin': list(map(float, frame.ImagePositionPatient)),
                    'Spacing': list(map(float, frame.PixelSpacing)),
                    'Thickness': float(frame.SliceThickness) if frame.SliceThickness else 1.0,
                })
            except AttributeError as error:
                print('Warning in file {} {}'.format(fileName, error),
                      file=sys.stderr)
        return sorted(frames, key=lambda frame: frame['Location'])

    def PrepareStringArray(self, frames):
        array = vtk.vtkStringArray()
        for frame in frames:
            array.InsertNextValue(frame['FileName'])
        return array

    def SetFileName(self, fileName):
        self.FileNameList = [fileName]

    def SetFileNames(self, fileNames):
        self.FileNameList = fileNames

    def SetDirectoryName(self, directoryName):
        self.FileNameList = [os.path.join(directoryName, fileName)
                             for fileName in os.listdir(directoryName)]

    def Update(self):
        frames = self.PrepareDICOMFrames(self.FileNameList)
        array = self.PrepareStringArray(frames)
        vtk.vtkDICOMImageReader.SetFileNames(self, array)
        self.SetDataExtent(0, frames[0]['Dimensions'][0] - 1,
                           0, frames[0]['Dimensions'][1] - 1,
                           0, len(frames) - 1)
        self.SetDataSpacing(frames[0]['Spacing'] + [frames[0]['Thickness']])
        self.SetDataOrigin(frames[0]['Origin'])
        vtk.vtkDICOMImageReader.Update(self)

