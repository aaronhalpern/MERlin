from typing import List
from typing import Union
import numpy as np
import pandas as pd
import time
from skimage import registration
from skimage import transform
from skimage import feature

import cv2

from merlin.core import analysistask
from merlin.util import aberration


class Warp(analysistask.ParallelAnalysisTask):

    """
    An abstract class for warping a set of images so that the corresponding
    pixels align between images taken in different imaging rounds.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'write_fiducial_images' not in self.parameters:
            self.parameters['write_fiducial_images'] = False
        if 'write_aligned_images' not in self.parameters:
            self.parameters['write_aligned_images'] = False

        self.writeAlignedFiducialImages = self.parameters[
                'write_fiducial_images']

    def get_aligned_image_set(
            self, fov: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the set of transformed images for the specified fov.

        Args:
            fov: index of the field of view
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 4-dimensional numpy array containing the aligned images. The
                images are arranged as [channel, zIndex, x, y]
        """
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zIndexes = range(len(self.dataSet.get_z_positions()))
        return np.array([[self.get_aligned_image(fov, d, z, chromaticCorrector)
                          for z in zIndexes] for d in dataChannels])

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the specified transformed image

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        inputImage = self.dataSet.get_raw_image(
            dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        transformation = self.get_transformation(fov, dataChannel)
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            return transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            return transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)

    def _process_transformations(self, transformationList, fov) -> None:
        """
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        if self.parameters['write_aligned_images']:
            zPositions = self.dataSet.get_z_positions()

            imageDescription = self.dataSet.analysis_tiff_description(
                    len(zPositions), len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    for z in zPositions:
                        inputImage = self.dataSet.get_raw_image(x, fov, z)
                        transformedImage = transform.warp(
                                inputImage, t, preserve_range=True) \
                            .astype(inputImage.dtype)
                        outputTif.save(
                                transformedImage,
                                photometric='MINISBLACK',
                                metadata=imageDescription)

        if self.writeAlignedFiducialImages:

            fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_fiducial_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                        .astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=fiducialImageDescription)

        self._save_transformations(transformationList, fov)

    def _save_transformations(self, transformationList: List, fov: int) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList), 'offsets',
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations')

    def get_transformation(self, fov: int, dataChannel: int=None
                            ) -> Union[transform.EuclideanTransform,
                                 List[transform.EuclideanTransform]]:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformationMatrices = self.dataSet.load_numpy_analysis_result(
            'offsets', self, resultIndex=fov, subdirectory='transformations')
        if dataChannel is not None:
            return transformationMatrices[dataChannel]
        else:
            return transformationMatrices


class FiducialCorrelationWarp(Warp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassSigma = self.parameters['highpass_sigma']
        highPassFilterSize = int(2 * np.ceil(2 * highPassSigma) + 1)
        inputImage = cv2.medianBlur(inputImage, ksize = 3)

        return inputImage.astype(float) - cv2.GaussianBlur(
            inputImage, (highPassFilterSize, highPassFilterSize),
            highPassSigma, borderType=cv2.BORDER_REPLICATE)

    def _run_analysis(self, fragmentIndex: int):
        # TODO - this can be more efficient since some images should
        # use the same alignment if they are from the same imaging round
        fixedImage = self._filter(
            self.dataSet.get_fiducial_image(0, fragmentIndex))
        offsets = [registration.phase_cross_correlation(
            fixedImage,
            self._filter(self.dataSet.get_fiducial_image(x, fragmentIndex)),
            upsample_factor = 100)[0] for x in
                   self.dataSet.get_data_organization().get_data_channels()]
        transformations = [transform.SimilarityTransform(
            translation=[-x[1], -x[0]]) for x in offsets]
        self._process_transformations(transformations, fragmentIndex)

class FiducialCorrelationWarp3D(FiducialCorrelationWarp):

    """
    An analysis task that warps a set of images taken in different imaging
    rounds based on the crosscorrelation between fiducial images.
    
    
    General plan - there are three corrections applied
    The first correction is that every stack is corrected for the piezo induced drift
        This uses a calibration of the z-stage position from the .off file
    The second correction is the XY registration from the fiducial bead frame
    The third correction is using an XYZ registration of the fiducial3D bead stacks
    
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 4096

    def get_estimated_time(self):
        return 5
        
    def get_piezo_corrected_frame(self,
                                fov: int,
                                dataChannel: int,
                                zIndex: int,
                                chromaticCorrector: aberration.ChromaticCorrector=None
                                ) -> np.ndarray:
        """Get the specified image corrected for piezo drift in XY
        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        
        inputImage = self.dataSet.get_raw_image(dataChannel, fov, self.dataSet.z_index_to_position(zIndex))
        inputImage_zstage_position = self.dataSet.get_raw_image_zstage_positions(dataChannel, fov)[zIndex]

        #### make sure this is correct
        # these are the interpolation functions - note the negative sign
        x_correction = -self.dataSet.piezo_xshift_function(self.dataSet.z_index_to_position(zIndex),
                                                           inputImage_zstage_position)
        y_correction = -self.dataSet.piezo_yshift_function(self.dataSet.z_index_to_position(zIndex),
                                                           inputImage_zstage_position)

        transformation = transform.SimilarityTransform(translation=[x_correction, y_correction])
        
        return transform.warp(inputImage, transformation, 
            preserve_range=True).astype(inputImage.dtype)                         

    def get_piezo_corrected_fiducial3D_stack(self, dataChannel, fragmentIndex):
        # this will return a piezo corrected stack of the fiducial3D image 
        # the assumption here is that the first frame index is the beads on the coverglass surface
        
        stack = self.dataSet.get_fiducial3D_stack(dataChannel, fragmentIndex)


        stack_zpositions = self.dataSet.get_data_organization().get_fiducial3D_stack_frame_zPos(
                                                            dataChannel)
        stack_zstage_positions = self.dataSet.get_fiducial_image_zstage_positions(dataChannel, 
                                                            fragmentIndex)

        x_correction = -self.dataSet.piezo_xshift_function(stack_zpositions,
                                                           stack_zstage_positions)
        y_correction = -self.dataSet.piezo_yshift_function(stack_zpositions,
                                                           stack_zstage_positions)

        transforms = [transform.SimilarityTransform(translation=[x, y]) for x,y in zip(x_correction,y_correction)]
        
        for i in range(len(stack)):
            stack[i] = transform.warp(stack[i], transforms[i], preserve_range=True).astype(stack.dtype)
        
        return stack
    
    # this is standard MERFISH registration
    # should not need to correct this for piezo since we correct to the fiducial frame postion anyways
    def _find_2D_offsets(self, fragmentIndex: int):
        
        fixedImage = self._filter(
                self.dataSet.get_fiducial_image(0, fragmentIndex))
        
        # phase cross cor returns Y X shifts
        offsets = [registration.phase_cross_correlation(
            fixedImage,
            self._filter(self.dataSet.get_fiducial_image(x, fragmentIndex)),
            upsample_factor = 100)[0] for x in
                   self.dataSet.get_data_organization().get_data_channels()]
        
        # should be Y X order
        offsets2D = [[x[0], x[1]] for x in offsets]
        return offsets2D
    
    def _find_offsets_from_3D_stacks(self, fragmentIndex: int):
        # this is for registration of a bead stack at the top of a 3D tissue
        # first register the zero plane
        fixedImage = self.get_piezo_corrected_fiducial3D_stack(0, fragmentIndex)
        
        offsets3D_base = []
        offsets3D = []
        
        for dataChannel in self.dataSet.get_data_organization().get_data_channels():
            movingImage = self.get_piezo_corrected_fiducial3D_stack(dataChannel, fragmentIndex)
            
            # 2D base offset this should be Y X
            offsets3D_base.append(registration.phase_cross_correlation(fixedImage[0], movingImage[0], upsample_factor = 100)[0])
            # 3D offsets.. this should be Z Y X
            offsets3D.append(registration.phase_cross_correlation(fixedImage[1:], movingImage[1:], upsample_factor = 100)[0])
            
        return offsets3D_base, offsets3D

    def _save_transformation_dataFrame(self, offsets2D, offsets3D_base, offsets3D, fragmentIndex: int):
        df = pd.DataFrame(columns = ['dataChannel','zPos','zPos_new','xshift','yshift'])
        dataChannels = self.dataSet.get_data_organization().get_data_channels()
        zPos_orig = self.dataSet.get_data_organization().get_z_positions()
        # assume all fiducial zpos are the same across channels
        fiducial3D_zpos = self.dataSet.get_data_organization().get_fiducial3D_stack_frame_zPos(0) 
        fiducial3D_zpos_center = np.mean(fiducial3D_zpos[1:]) # first frame is assumed to be zero

        # zip datachannel, 2d offsets, 3d base offsets, 3d stack offsets
        for dc, (off2D_y, off2D_x), (off3Db_y,off3Db_x), (off3D_z, off3D_y, off3D_x) in zip(dataChannels, offsets2D, offsets3D_base, offsets3D):
            df_temp = pd.DataFrame(columns = ['dataChannel','zPos','zPos_new','xshift','yshift'])
            # these are the 2d shift and a z-dependant shift from the 3d registration
            # make negative since we are shifting the fixed image

            #                   2d          3d - 3d base    * some percentage of the height
            df_temp['yshift'] = -off2D_y - (off3D_y-off3Db_y)*np.array(zPos_orig)/fiducial3D_zpos_center
            df_temp['xshift'] = -off2D_x - (off3D_x-off3Db_x)*np.array(zPos_orig)/fiducial3D_zpos_center
            df_temp['zPos'] = zPos_orig # original z pos just so its in the df
            # new zpos. shift is the direction to move the moving image
            # if zshift is negative, the moving image has expanded 
            # so our new zpos should be larger hence the negative below to make the ratio larger
            df_temp['zPos_new'] = np.array(zPos_orig) * (fiducial3D_zpos_center - off3D_z)/fiducial3D_zpos_center
            # include the datachannel
            df_temp['dataChannel'] = dc
            # include raw z shift for troubleshooting purposes
            df_temp['zshift'] = off3D_z

            df = pd.concat([df, df_temp], ignore_index=True)
        
        self.dataSet.save_dataframe_to_csv(df,
                                           'transformation_table',
                                           self.get_analysis_name(),
                                           resultIndex=fragmentIndex,
                                           subdirectory='transformations')                   
    
    def get_transformation_table(self, fov: int) -> pd.DataFrame:
        """Get the transformations for aligning images for the specified field
        of view.

        Args:
            fov: the fov to get the transformations for.
            dataChannel: the index of the data channel to get the transformation
                for. If None, then all data channels are returned.
        Returns:
            a EuclideanTransform if dataChannel is specified or a list of
                EuclideanTransforms for all dataChannels if dataChannel is
                not specified.
        """
        transformation_table = self.dataSet.load_dataframe_from_csv(
            'transformation_table', self, resultIndex=fov, subdirectory='transformations')
            
        return transformation_table

    def get_aligned_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector=None
    ) -> np.ndarray:
        """Get the specified transformed image
        corrected using 3d beads

        Args:
            fov: index of the field of view
            dataChannel: index of the data channel
            zIndex: index of the z position
            chromaticCorrector: the ChromaticCorrector to use to chromatically
                correct the images. If not supplied, no correction is
                performed.
        Returns:
            a 2-dimensional numpy array containing the specified image
        """
        df = self.get_transformation_table(fov)
        zPos = self.dataSet.z_index_to_position(zIndex)
        df = df[(df['dataChannel'] == dataChannel) & 
                     (df['zPos'] == zPos)]
        
        xshift = df['xshift'].values[0]
        yshift = df['yshift'].values[0]
        zPos_new = df['zPos_new'].values[0]

        zPos_all = np.array(self.dataSet.get_z_positions())
        
        if zPos_new > np.amax(zPos_all):
            zPos_new = np.amax(zPos_all)
        
        # interpolate the two nearest frames
        zPos_nearest = zPos_all[np.abs(zPos_all - zPos_new).argsort()[0:2]] # take two nearest zpos
        zPos_nearest_distances = np.abs(zPos_new - zPos_nearest) # find distance
        weights = 1 - zPos_nearest_distances/np.sum(zPos_nearest_distances) # get weighting factor
        
        images_nearest = [self.dataSet.get_raw_image(dataChannel, fov, z) for z in zPos_nearest]
        
        # interpolated image
        inputImage = (images_nearest[0] * weights[0] +
                      images_nearest[1] * weights[1]).astype(
                      images_nearest[0].dtype) 
                      
        transformation = transform.SimilarityTransform(
            translation=[xshift, yshift])

        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            return transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            return transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)
                                  

    def _process_transformations(self, fov) -> None:
        """
        Process the transformations determined for a given fov. 

        The list of transformation is used to write registered images and 
        the transformation list is archived.

        Args:
            transformationList: A list of transformations that contains a
                transformation for each data channel. 
            fov: The fov that is being transformed.
        """

        dataChannels = self.dataSet.get_data_organization().get_data_channels()

        if self.parameters['write_aligned_images']:
            zPositions = self.dataSet.get_z_positions()

            imageDescription = self.dataSet.analysis_tiff_description(
                    len(zPositions), len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_images', fov) as outputTif:
                for x in dataChannels:
                    for z in zPositions:
                        #print('aligning channel {} zpos {}'.format(x, z))
                        transformedImage = self.get_aligned_image(fov, x, 
                                                self.dataSet.position_to_z_index(z))
                        outputTif.save(
                                transformedImage,
                                photometric='MINISBLACK',
                                metadata=imageDescription)

        # this should be unchanged from normal merlin
        if self.writeAlignedFiducialImages:

            transformationList = self.get_transformation(fov)

            fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_fiducial_images', fov) as outputTif:
                for t, x in zip(transformationList, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformedImage = transform.warp(
                            inputImage, t, preserve_range=True) \
                        .astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=fiducialImageDescription)
                    

    def _run_analysis(self, fragmentIndex: int):
        print('running 2d registration')
        offsets2D = self._find_2D_offsets(fragmentIndex)

        # just save the 2D transformation like normal merlin
        transformations2D = [transform.SimilarityTransform(
            translation=[-x[1], -x[0]]) for x in offsets2D]
        
        self._save_transformations(transformations2D, fragmentIndex)
        
        # now do 3D registration
        print('running 3d registration')
        offsets3D_base, offsets3D = self._find_offsets_from_3D_stacks(fragmentIndex)
        
        self._save_transformation_dataFrame(offsets2D, offsets3D_base, offsets3D, fragmentIndex)

        self._process_transformations(fragmentIndex)