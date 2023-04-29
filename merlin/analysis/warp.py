from typing import List
from typing import Union
import numpy as np
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
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'zstage_reference_position' not in self.parameters:
            self.parameters['zstage_reference_position'] = 50 # 50 um starting zstage position

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 4096

    def get_estimated_time(self):
        return 5

    def get_piezo_corrected_fiducial3D_stack(self, dataChannel, fragmentIndex):
        
        stack = self.dataSet.get_fiducial3D_stack(dataChannel, fragmentIndex)
        stack_indices = self.dataSet.get_data_organization().get_fiducial3D_stack_frame_indices(dataChannel)
        
        zstage_positions_all = self.dataSet.get_fiducial_image_zstage_positions(dataChannel, fragmentIndex)
        zstage_positions_stack = zstage_positions_all[stack_indices]
        zstage_position_ref = zstage_positions_all[self.dataSet.get_data_organization().get_fiducial3D_base_frame_index(0)]

        x_correction = -np.polyval(self.dataSet.piezo_xshift_coeffs, zstage_positions) + np.polyval(self.dataSet.piezo_xshift_coeffs, zstage_position_ref)
        y_correction = -np.polyval(self.dataSet.piezo_yshift_coeffs, zstage_positions) + np.polyval(self.dataSet.piezo_yshift_coeffs, zstage_position_ref)

        transforms = [transform.SimilarityTransform(translation=[x, y]) for x,y in zip(x_correction,y_correction)]
        
        for i in range(len(stack)):
            stack[i] = transform.warp(stack[i], transformation, preserve_range=True).astype(stack.dtype)
        
        return stack
        
    def get_piezo_corrected_fiducial3D_base_image(self, dataChannel, fragmentIndex):
        
        stack = self.dataSet.get_fiducial3D_stack(dataChannel, fragmentIndex)
        stack_indices = self.dataSet.get_data_organization().get_fiducial3D_stack_frame_indices(dataChannel)
        
        zstage_positions_all = self.dataSet.get_fiducial_image_zstage_positions(dataChannel, fragmentIndex)
        zstage_positions_stack = zstage_positions_all[stack_indices]
        zstage_position_ref = zstage_positions_all[self.dataSet.get_data_organization().get_fiducial3D_base_frame_index(0)]

        x_correction = -np.polyval(self.dataSet.piezo_xshift_coeffs, zstage_positions) + np.polyval(self.dataSet.piezo_xshift_coeffs, zstage_position_ref)
        y_correction = -np.polyval(self.dataSet.piezo_yshift_coeffs, zstage_positions) + np.polyval(self.dataSet.piezo_yshift_coeffs, zstage_position_ref)

        transforms = [transform.SimilarityTransform(translation=[x, y]) for x,y in zip(x_correction,y_correction)]
        
        for i in range(len(stack)):
            stack[i] = transform.warp(stack[i], transformation, preserve_range=True).astype(stack.dtype)
        
        return stack

base = self.dataSet.get_fiducial3D_base_image(dataChannel, fragmentIndex)
base_index = self.dataSet.get_data_organization().get_fiducial3D_base_frame_index(dataChannel)


    def _find_2D_offsets(self, fragmentIndex: int):
        # this is standard MERFISH registration
        fixedImage = self._filter(
                self.dataSet.get_fiducial_image(0, fragmentIndex))
              
        offsets = [registration.phase_cross_correlation(
            fixedImage,
            self._filter(self.dataSet.get_fiducial_image(x, fragmentIndex)),
            upsample_factor = 100)[0] for x in
                   self.dataSet.get_data_organization().get_data_channels()]
                   
        #transformations = [transform.SimilarityTransform(
        #    translation=[-x[1], -x[0]]) for x in offsets]
        
        # saved as X Y
        transformations = [[-x[1], -x[0]] for x in offsets]
        return transformations
    
    def _find_2D_base_offsets_from_3D_stacks(self, fragmentIndex: int):
        # this is for registration of a bead stack at the top of a 3D tissue
        # first register the zero plane
        fixedImage = self._filter(
                self.dataSet.get_fiducial3D_base_image(0, fragmentIndex))
              
        offsets = [registration.phase_cross_correlation(
            fixedImage,
            self._filter(self.dataSet.get_fiducial3D_base_image(x, fragmentIndex)),
            upsample_factor = 100)[0] for x in
                   self.dataSet.get_data_organization().get_data_channels()]
                   
        #transformations = [transform.SimilarityTransform(
        #    translation=[-x[1], -x[0], 0], dimensionality = 3) \
        #        for x in offsets]
        
        # saved as X, Y
        transformations = [[-x[1], -x[0], 0] for x in offsets]
        
        return transformations
    
    def _find_3D_offsets_from_3D_stacks(self, fragmentIndex: int):
        # this is for registration of a bead stack at the top of a 3D tissue
        # first register the zero plane
        fixedImage = self.dataSet.get_fiducial3D_stack(0, fragmentIndex)
        offsets = [registration.phase_cross_correlation(fixedImage,
            self.dataSet.get_fiducial3D_stack(x, fragmentIndex),
            upsample_factor = 100)[0] for x in
            self.dataSet.get_data_organization().get_data_channels()]

        # will need skimage > 0.17 for XYZ similarity transform
        # given by translation = x, y, z
        #transformations = [transform.SimilarityTransform(
        #    translation=[-x[2], -x[1], -x[0]], dimensionality = 3 ) for x in offsets]
        
        # saved as X, Y, Z
        transformations = [[-x[2], -x[1], -x[0]] for x in offsets]
        return transformations

    def _find_piezo_correction(self, fragmentIndex: int):
        # something something something
        pass


    def _save_transformations(self, transformationList: List, fov: int, name: str) -> None:
        self.dataSet.save_numpy_analysis_result(
            np.array(transformationList), name,
            self.get_analysis_name(), resultIndex=fov,
            subdirectory='transformations')
            

    def _run_analysis(self, fragmentIndex: int):
        transforms_2D = self._find_2D_offsets(fragmentIndex)
        self._save_transformations(transforms_2D, fragmentIndex, 'offsets2D')
        
        transforms_3D_base = self._find_2D_base_offsets_from_3D_stacks(fragmentIndex)
        self._save_transformations(transforms_3D_base, fragmentIndex, 'offsets3D_base')
        
        transforms_3D = self._find_3D_offsets_from_3D_stacks(fragmentIndex)
        self._save_transformations(transforms_3D, fragmentIndex, 'offsets3D')
        
        self._process_transformations(fragmentIndex)
    
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
        transformation_2D = self.dataSet.load_numpy_analysis_result(
            'offsets2D', self, resultIndex=fov, subdirectory='transformations')
            
        if dataChannel is not None:
            return transformation_2D[dataChannel]
        else:
            return transformation_2D

    def index_to_corrected_zPos(self, fov: int, dataChannel: int, zIndex: int) -> float:
        fiducial3D_zpos_mean = np.mean(self.dataSet.get_data_organization().get_fiducial3D_stack_frame_zPos(dataChannel))
   
        transformation_3D_zshift = self.dataSet.load_numpy_analysis_result(
            'offsets3D', self, resultIndex=fov, subdirectory='transformations')[dataChannel, 2]
        
        # zPos we want to get
        zPos = self.dataSet.z_index_to_position(zIndex)
        
        # zPos accounting for 3D registration
        # assumption here is that the base frame is at z = 0
        zPos_new = zPos * (fiducial3D_zpos_mean + transformation_3D_zshift)/fiducial3D_zpos_mean
        
        return zPos_new
    
    def index_to_corrected_zPos_image(self, fov: int, dataChannel: int, zIndex: int) -> np.ndarray:
    
        zPos_new = self.index_to_corrected_zPos(fov, dataChannel, zIndex)
        
        zPos_all = self.dataSet.get_z_positions()
        
        if zPos_new > np.amax(zPos_all):
            zPos_new = np.amax(zPos_all)
        
        # interpolate the two nearest frames
        zPos_nearest = zPos_all[np.abs(zPos_all - zPos_new).argsort()[0:2]] # take two nearest zpos
        zPos_nearest_distances = np.abs(zPos_new - zPos_nearest) # find distance
        weights = 1 - zPos_nearest_distances/np.sum(zPos_nearest_distance) # get weighting factor
        
        images_nearest = [self.dataSet.get_raw_image(dataChannel, fov, z) for z in zPos_nearest]
        
        return (images_nearest[0] * weights[0] + images_nearest[1] * weights[1]).astype(images_nearest[0]) # interpolated image
        
    def index_to_XY_correction_transform(self, fov: int, dataChannel: int, zIndex: int) -> np.ndarray:
        
        # fiducial plane
        offsets2D = self.dataSet.load_numpy_analysis_result(
            'offsets2D', self, resultIndex=fov, subdirectory='transformations')[dataChannel]

        transformation_3D_base = self.dataSet.load_numpy_analysis_result(
            'offsets3D_base', self, resultIndex=fov, subdirectory='transformations')[dataChannel] 
    
        transformation_3D = self.dataSet.load_numpy_analysis_result(
            'offsets3D', self, resultIndex=fov, subdirectory='transformations')[dataChannel] 
        
        fiducial3D_zpos_mean = np.mean(self.dataSet.get_data_organization().get_fiducial3D_stack_frame_zPos(dataChannel))
        
        offsets_3D_slope = (transformation_3D - transformation_3D_base)/fiducial3D_zpos_mean
        zPos_new = self.index_to_corrected_zPos(fov, dataChannel, zIndex)
        
        offsets_3D = offsets_3D_slope * zPos_new
        
        offsets = offsets2D + offsets3D
        
        return transform.SimilarityTransform(translation = offsets[0:2])

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

        inputImage = self.dataSet.index_to_corrected_zPos_image(dataChannel, fov, zIndex)
        transformation = self.index_to_XY_correction_transform(dataChannel, fov, zIndex)
        
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
                        transformedImage = self.get_aligned_image(fov, x, self.dataSet.position_to_z_index(z))
                        outputTif.save(
                                transformedImage,
                                photometric='MINISBLACK',
                                metadata=imageDescription)

        if self.writeAlignedFiducialImages:

            fiducialImageDescription = self.dataSet.analysis_tiff_description(
                    1, len(dataChannels))
                    
            transformation_2D = self.dataSet.load_numpy_analysis_result(
                'offsets2D', self, resultIndex=fov, subdirectory='transformations')

            with self.dataSet.writer_for_analysis_images(
                    self, 'aligned_fiducial_images', fov) as outputTif:
                for t, x in zip(transformation_2D, dataChannels):
                    inputImage = self.dataSet.get_fiducial_image(x, fov)
                    transformation = transform.SimilarityTransform(translation = [t[0], t[1]])
                    transformedImage = transform.warp(inputImage, transformation,
                        preserve_range=True).astype(inputImage.dtype)
                    outputTif.save(
                            transformedImage, 
                            photometric='MINISBLACK',
                            metadata=fiducialImageDescription)
