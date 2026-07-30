import os
import subprocess
import cv2
import numpy as np
import scipy as sp

from merlin.core import analysistask
from merlin.util import deconvolve
from merlin.util import aberration
from merlin.util import imagefilters
from merlin.data import codebook

from skimage import transform
from skimage import io

class Preprocess(analysistask.ParallelAnalysisTask):

    """
    An abstract class for preparing data for barcode calling.
    """

    def _image_name(self, fov):
        destPath = self.dataSet.get_analysis_subdirectory(
                self.analysisName, subdirectory='preprocessed_images')
        return os.sep.join([destPath, 'fov_' + str(fov) + '.tif'])

    def get_pixel_histogram(self, fov=None):
        if fov is not None:
            return self.dataSet.load_numpy_analysis_result(
                'pixel_histogram', self.analysisName, fov, 'histograms')

        pixelHistogram = np.zeros(self.get_pixel_histogram(
                self.dataSet.get_fovs()[0]).shape)
        for f in self.dataSet.get_fovs():
            pixelHistogram += self.get_pixel_histogram(f)

        return pixelHistogram

    def _save_pixel_histogram(self, histogram, fov):
        self.dataSet.save_numpy_analysis_result(
            histogram, 'pixel_histogram', self.analysisName, fov, 'histograms')

class CAREPreprocess(Preprocess):
    def __init__(self, dataSet, parameters=None, analysisName=None):
            super().__init__(dataSet, parameters, analysisName)

            # do lazy import since this is slow to load...
            try:
                from csbdeep.models import CARE
            except ImportError:
                raise ImportError('***CARE package (csbdeep.models.CARE) not found***')


            if 'CARE_model_directory' not in self.parameters:
                raise ValueError('CARE model path not in parameters')

            if 'codebook_index' not in self.parameters:
                self.parameters['codebook_index'] = 0
            if 'write_preprocessed_images' not in self.parameters:
                self.parameters['write_preprocessed_images'] = False
            if 'highpass_sigma' not in self.parameters:
                self.parameters['highpass_sigma'] = 3
            # turn of save pixel histogram - it can be time consuming for CARE
            # will assume initial scale factors are = 1 in Optimization
            if 'save_pixel_histogram' not in self.parameters:
                self.parameters['save_pixel_histogram'] = False
            if 'write_preprocessed_FOV' not in self.parameters:
                self.parameters['write_preprocessed_FOV'] = [-1]
            
            self._highPassSigma = self.parameters['highpass_sigma']

            self.warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])
        
            # is this a good way to bring in the model?
            model_basedir, model_name = os.path.split(self.parameters['CARE_model_directory'])
            
            self.model = CARE(config = None,
                             name = model_name,
                             basedir= model_basedir)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 4096

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for zIndex in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                    for b in self.get_codebook().get_bit_names()])

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex,
                                                    chromaticCorrector)
        return self._preprocess_image(inputImage)
    
    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        outputImage = self.model.predict(inputImage, 'YX')
        outputImage = self._high_pass_filter(outputImage)
        return outputImage.astype(np.float32) # switching back to float32 will it eat up memory?
        
    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                self._highPassSigma)
        return hpImage #.astype(np.float32) # does this need to be cast?
    
    def _run_analysis(self, fragmentIndex):
    
        if self.parameters['write_preprocessed_images']:
            if self.parameters['write_preprocessed_FOV'] == [-1]:
                self.parameters['write_preprocessed_FOV'] = self.dataSet.get_fovs()
            
        if self.parameters['save_pixel_histogram'] or (fragmentIndex in self.parameters['write_preprocessed_FOV']):
        
            warpTask = self.dataSet.load_analysis_task(
                    self.parameters['warp_task'])

            histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
            pixelHistogram = np.zeros(
                    (self.get_codebook().get_bit_count(), len(histogramBins)-1))

            # this currently only is to calculate the pixel histograms in order
            # to estimate the initial scale factors. This is likely unnecessary

            with self.dataSet.writer_for_analysis_images(
                     self.analysisName, 'preprocessed_images', fragmentIndex) as outputTif:

                for bi, b in enumerate(self.get_codebook().get_bit_names()):
                    dataChannel = self.dataSet.get_data_organization()\
                            .get_data_channel_for_bit(b)
                    for i in range(len(self.dataSet.get_z_positions())):
                        inputImage = warpTask.get_aligned_image(
                                fragmentIndex, dataChannel, i)
                        outputImage = self._preprocess_image(inputImage)

                        pixelHistogram[bi, :] += np.histogram(
                                outputImage, bins=histogramBins)[0]
                        
                        outputTif.save(outputImage,photometric='MINISBLACK')

            self._save_pixel_histogram(pixelHistogram, fragmentIndex)

class DeconvolutionPreprocess(Preprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 3
        if 'decon_sigma' not in self.parameters:
            self.parameters['decon_sigma'] = 2
        if 'decon_filter_size' not in self.parameters:
            self.parameters['decon_filter_size'] = \
                int(2 * np.ceil(2 * self.parameters['decon_sigma']) + 1)
        if 'decon_iterations' not in self.parameters:
            self.parameters['decon_iterations'] = 20
        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0
        
        # add some options to save preprocessed images
        if 'write_preprocessed_images' not in self.parameters:
            self.parameters['write_preprocessed_images'] = False                
        if 'write_preprocessed_FOV' not in self.parameters:
            self.parameters['write_preprocessed_FOV'] = list(range(self.fragment_count()))
        if 'save_pixel_histogram' not in self.parameters:
            self.parameters['save_pixel_histogram'] = True

        self._highPassSigma = self.parameters['highpass_sigma']
        self._deconSigma = self.parameters['decon_sigma']
        self._deconIterations = self.parameters['decon_iterations']

        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for zIndex in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                    for b in self.get_codebook().get_bit_names()])

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        inputImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex,
                                                     chromaticCorrector)
        return self._preprocess_image(inputImage)

    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                self._highPassSigma)
        return hpImage.astype(np.float32)

    def _run_analysis(self, fragmentIndex):
            
        if self.parameters['save_pixel_histogram'] or (fragmentIndex in self.parameters['write_preprocessed_FOV']):
    
            warpTask = self.dataSet.load_analysis_task(
                    self.parameters['warp_task'])

            histogramBins = np.arange(0, np.iinfo(np.uint16).max, 1)
            pixelHistogram = np.zeros(
                    (self.get_codebook().get_bit_count(), len(histogramBins)-1))

                # this currently only is to calculate the pixel histograms in order
                # to estimate the initial scale factors. This is likely unnecessary?
                
            with self.dataSet.writer_for_analysis_images(
                     self.analysisName, 'preprocessed_images', fragmentIndex) as outputTif:
                
                for bi, b in enumerate(self.get_codebook().get_bit_names()):
                    dataChannel = self.dataSet.get_data_organization()\
                            .get_data_channel_for_bit(b)
                    
                    for i in range(len(self.dataSet.get_z_positions())):
                        inputImage = warpTask.get_aligned_image(
                                fragmentIndex, dataChannel, i)
                        deconvolvedImage = self._preprocess_image(inputImage)

                        pixelHistogram[bi, :] += np.histogram(
                                deconvolvedImage, bins=histogramBins)[0]
                        
                        if self.parameters['write_preprocessed_images']:
                            outputTif.save(deconvolvedImage, photometric='MINISBLACK')

            self._save_pixel_histogram(pixelHistogram, fragmentIndex)

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters['decon_filter_size']

        filteredImage = self._high_pass_filter(inputImage)
        # bypass decon
        if self._deconIterations > 0:
            deconvolvedImage = deconvolve.deconvolve_lucyrichardson(
                filteredImage, deconFilterSize, self._deconSigma,
                self._deconIterations).astype(np.uint16)
            return deconvolvedImage
        else:
            return filteredImage

class DeconvolutionPreprocessDW(Preprocess):
    
    def __init__(self, dataSet, parameters=None, analysisName=None):
            super().__init__(dataSet, parameters, analysisName)

            if 'codebook_index' not in self.parameters:
                self.parameters['codebook_index'] = 0
            if 'highpass_sigma' not in self.parameters:
                self.parameters['highpass_sigma'] = 3
            # turn off save pixel histogram?
            # this will assume initial scale factors are = 1 in Optimization
            if 'save_pixel_histogram' not in self.parameters:
                self.parameters['save_pixel_histogram'] = True
            if 'histogram_bin_max' not in self.parameters:
                self.parameters['histogram_bin_max'] = 10000000
                # due to the way scale factors are calculated
                # we need to bin at integer amounts
                # for float 32 we would need ridiculous number of bins...
                # not sure the best way to overcome this for now...

            self._highPassSigma = self.parameters['highpass_sigma']

            self.warpTask = self.dataSet.load_analysis_task(
                self.parameters['warp_task'])
            
            # here are params for deconwolf

            if 'dw_path' not in self.parameters: 
                self.parameters['dw_path'] = 'dw' # assumes dw is in path
                
            if 'iterations' not in self.parameters:
                self.parameters['iterations'] = 15

            if 'use_gpu' not in self.parameters:
                self.parameters['use_gpu'] = True

            if 'overwrite' not in self.parameters:
                # will enable resumable decon
                self.parameters['overwrite'] = False 
            
            """
            # TURNOFF TILING it seems incompatible with --float...
            if 'tilesize' not in self.parameters:
                self.parameters['tilesize'] = 1024

            if 'tilepad' not in self.parameters:
                self.parameters['tilepad'] = 128
            """

            # find all the wavelengths and channels
            # but only for bits in codebook

            self.bits = self.get_codebook().get_bit_names()
            self.channels = [self.dataSet.get_data_organization().get_data_channel_for_bit(b) 
                             for b in self.bits]
            self.wavelengths = [self.dataSet.get_data_organization().get_data_channel_color(channel) 
                            for channel in self.channels]
            self.wavelengths = set(self.wavelengths)

            # make a dictionary of the PSF paths
            if 'psf_directory' in self.parameters:
                
                self.PSF_paths = {}
                base_path = self.parameters['psf_directory']
                for wavelength in self.wavelengths:
                    fpath = os.path.join(base_path, f'PSF_{wavelength}.tif')
                    if os.path.isfile(fpath):
                        self.PSF_paths[wavelength] = fpath
                    else:
                        raise ValueError(f'could not find PSF_{wavelength}.tif')
            else:
                    raise ValueError(f'no PSFs found')
            
            if 'remove_conventional_image' not in self.parameters:
                self.parameters['remove_conventional_image'] = True # saves space
            
    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 16384

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return [self.parameters['warp_task']]

    def get_codebook(self) -> codebook.Codebook:
        return self.dataSet.get_codebook(self.parameters['codebook_index'])

    def get_raw_image_name(self, dataChannel: int) -> str:
        return f"channel_{dataChannel}_fov_"

    def get_raw_image_path(self, dataChannel: int, fov: int) -> str:
        imageBaseName = self.get_raw_image_name(dataChannel)
        return self.dataSet._analysis_image_name(
                self.analysisName, imageBaseName, fov)
    
    def get_dw_image_path(self, dataChannel: int, fov: int) -> str:
        imageBaseName = "dw_" + self.get_raw_image_name(dataChannel)
        return self.dataSet._analysis_image_name(
                self.analysisName, imageBaseName, fov)

    def get_processed_image_set(
            self, fov, zIndex: int = None,
            chromaticCorrector: aberration.ChromaticCorrector = None
    ) -> np.ndarray:
        
        if zIndex is None:
            return np.array([[self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                for zIndex in range(len(self.dataSet.get_z_positions()))]
                for b in self.get_codebook().get_bit_names()])
        else:
            return np.array([self.get_processed_image(
                fov, self.dataSet.get_data_organization()
                    .get_data_channel_for_bit(b), zIndex, chromaticCorrector)
                    for b in self.get_codebook().get_bit_names()])

    def get_processed_image(
            self, fov: int, dataChannel: int, zIndex: int,
            chromaticCorrector: aberration.ChromaticCorrector = None) -> np.ndarray:

        imagePath = self.get_dw_image_path(dataChannel, fov)
        inputImage = self.dataSet.load_image(imagePath, zIndex, transform = False) # images are already transformed

        transformation = self.warpTask.get_transformation(fov, dataChannel)

        # this is from the warp class
        if chromaticCorrector is not None:
            imageColor = self.dataSet.get_data_organization()\
                            .get_data_channel_color(dataChannel)
            outputImage =  transform.warp(chromaticCorrector.transform_image(
                inputImage, imageColor), transformation, preserve_range=True
                ).astype(inputImage.dtype)
        else:
            outputImage = transform.warp(inputImage, transformation,
                                  preserve_range=True).astype(inputImage.dtype)

        # here is where the high pass happens
        outputImage = self._high_pass_filter(outputImage)

        return outputImage
        
    def _high_pass_filter(self, inputImage: np.ndarray) -> np.ndarray:
        highPassFilterSize = int(2 * np.ceil(2 * self._highPassSigma) + 1)
        hpImage = imagefilters.high_pass_filter(inputImage,
                                                highPassFilterSize,
                                                self._highPassSigma)
        return hpImage.astype(np.float32)
    
    def _run_analysis(self, fragmentIndex):

        for bi, b in enumerate(self.bits): # this will only do bits in the codebook
            dataChannel = self.dataSet.get_data_organization().get_data_channel_for_bit(b)
            wavelength = self.dataSet.get_data_organization().get_data_channel_color(dataChannel)
            
            #  check if the channel is already deconvolved
            if self.parameters['overwrite'] == False:
                dw_image_path = self.get_dw_image_path(dataChannel, fragmentIndex)
                dw_image_name = os.path.split(dw_image_path)[-1]
                if os.path.exists(dw_image_path):
                    print(f'found {dw_image_name}, skipping dw on channel {dataChannel}')
                    continue # skip the loop
            
            # write the raw image zstacks to disk
            with self.dataSet.writer_for_analysis_images(
                     self.analysisName,
                     self.get_raw_image_name(dataChannel), 
                     fragmentIndex) as outputTif:
                
                for zPosition in self.dataSet.get_z_positions():
                        frame = self.dataSet.get_raw_image(dataChannel, fragmentIndex, zPosition)
                        outputTif.save(frame, photometric='MINISBLACK')

            # this is the path of the image that was just saved
            inputImagePath = self.get_raw_image_path(dataChannel, fragmentIndex)

            # compose the dw command
            dw_command = []
            dw_command.append(self.parameters['dw_path'])
            dw_command.append('--iter')
            dw_command.append(str(self.parameters['iterations']))
            if self.parameters['use_gpu']:
                dw_command.append('--gpu')
            if self.parameters['overwrite']:
                dw_command.append('--overwrite')
            dw_command.append('--float') # this may be important so the image is not scaled funny...

            # turning off tiling, it seems incompatible with the --float option
            # also scale does not seem to work...
            #dw_command.append('--tilesize')
            #dw_command.append(str(self.parameters['tilesize']))
            #dw_command.append('--tilepad')
            #dw_command.append(str(self.parameters['tilepad']))
            #dw_command.append('--out') # don't use
            #dw_command.append(outputImagePath) # don't use
            
            dw_command.append(inputImagePath)
            dw_command.append(self.PSF_paths[wavelength])

            if True: # for troubleshooting
                print('running dw command: ' + ' '.join(dw_command))

            # run dw
            try:
                ret = subprocess.run(dw_command, check = True)
            except subprocess.CalledProcessError as e:
                raise Exception(f'dw error on channel {dataChannel} fov {fragmentIndex}')
                # I believe this should get caught by the analysistask

            #if ret.returncode != 0:
            #    raise Exception(f'dw error on channel {dataChannel} fov {fragmentIndex}')

            # remove the conventional image?
            if self.parameters['remove_conventional_image']:
                os.remove(inputImagePath)
        
        # calculate pixel histogram?
        if self.parameters['save_pixel_histogram']:

            # see note in params about histogram bin max
            # annoying to calculate this for float32 images
            # but may be necessary since thats what dw should output

            histogramBins = np.arange(0, self.parameters['histogram_bin_max'], 1)
            pixelHistogram = np.zeros(
                            (len(self.bits),
                             len(histogramBins)-1), np.int32)

            for bi, b in enumerate(self.bits): # only do bits in the codebook
                dataChannel = self.dataSet.get_data_organization().get_data_channel_for_bit(b)

                imagePath = self.get_dw_image_path(dataChannel, fragmentIndex)
                dw_image = io.imread(imagePath)
                preprocessedImage = np.array([self._high_pass_filter(im) for im in dw_image])
                # since this is a lot of data and a lot of bins to histogram
                # do a max projection
                preprocessedImage = np.amax(preprocessedImage, axis = 0)

                # finally do histogram
                h, _ = np.histogram(preprocessedImage, bins=histogramBins)

                # write that to the histogram file
                pixelHistogram[bi, :] = h

            self._save_pixel_histogram(pixelHistogram, fragmentIndex)

    def _save_pixel_histogram(self, histogram, fov):
        # get a save path
        savePath = self.dataSet._analysis_result_save_path(
                'pixel_histogram', 
                self.analysisName, 
                fov, 
                'histograms',
                '.npz')
        # convert to spares matrix
        sparse_matrix = sp.sparse.csr_matrix(histogram)
        sp.sparse.save_npz(savePath, sparse_matrix)

class DeconvolutionPreprocessGuo(DeconvolutionPreprocess):

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # Check for 'decon_iterations' in parameters instead of
        # self.parameters as 'decon_iterations' is added to
        # self.parameters by the super-class with a default value
        # of 20, but we want the default value to be 2.
        if 'decon_iterations' not in parameters:
            self.parameters['decon_iterations'] = 2

        self._deconIterations = self.parameters['decon_iterations']

    def _preprocess_image(self, inputImage: np.ndarray) -> np.ndarray:
        deconFilterSize = self.parameters['decon_filter_size']

        filteredImage = self._high_pass_filter(inputImage)
        deconvolvedImage = deconvolve.deconvolve_lucyrichardson_guo(
            filteredImage, deconFilterSize, self._deconSigma,
            self._deconIterations).astype(np.uint16)
        return deconvolvedImage
