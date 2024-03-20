import pandas
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import label
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.util import imagefilters


class SumSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that calculates the signal intensity within the boundaries
    of a cell for all rounds not used in the codebook, useful for measuring
    RNA species that were stained individually.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'apply_highpass' not in self.parameters:
            self.parameters['apply_highpass'] = False
        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 5
        if 'z_index' not in self.parameters:
            self.parameters['z_index'] = 0

        if self.parameters['z_index'] >= len(self.dataSet.get_z_positions()):
            raise analysistask.InvalidParameterException(
                'Invalid z_index specified for %s. (%i > %i)'
                % (self.analysisName, self.parameters['z_index'],
                   len(self.dataSet.get_z_positions())))

        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        if 'save_sequential_images' not in self.parameters:
            self.parameters['save_sequential_images'] = False

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _extract_signal(self, cells, inputImage, zIndex) -> pandas.DataFrame:
        cellCoords = []
        for cell in cells:
            regions = cell.get_boundaries()[zIndex]
            if len(regions) == 0:
                cellCoords.append([])
            else:
                pixels = []
                for region in regions:
                    coords = region.exterior.coords.xy
                    xyZip = list(zip(coords[0].tolist(), coords[1].tolist()))
                    pixels.append(np.array(
                                self.alignTask.global_coordinates_to_fov(
                                    cell.get_fov(), xyZip)))
                cellCoords.append(pixels)

        cellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))]
        mask = np.zeros(inputImage.shape, np.uint8)
        for i, cell in enumerate(cellCoords):
            cv2.drawContours(mask, cell, -1, i+1, -1)
        propsDict = {x.label: x for x in regionprops(mask, inputImage)}
        propsOut = pandas.DataFrame(
            data=[(propsDict[k].intensity_image.sum(),
                   propsDict[k].filled_area)
                  if k in propsDict else (0, 0)
                  for k in range(1, len(cellCoords) + 1)],
            index=cellIDs,
            columns=['Intensity', 'Pixels'])
        return propsOut

    def _get_sum_signal(self, fov, channels, zIndex):

        fTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])

        cells = sTask.get_feature_database().read_features(fov)

        signals = []
        for ch in channels:
            img = fTask.get_aligned_image(fov, ch, zIndex)
            if self.parameters['apply_highpass']:
                highPassSigma = self.parameters['highpass_sigma']
                highPassFilterSize = int(2 * np.ceil(3 * highPassSigma) + 1)
                img = imagefilters.high_pass_filter(img,
                                                    highPassFilterSize,
                                                    highPassSigma)
            signals.append(self._extract_signal(cells, img,
                                                zIndex).iloc[:, [0]])

        # adding num of pixels
        signals.append(self._extract_signal(cells, img, zIndex).iloc[:, [1]])

        compiledSignal = pandas.concat(signals, axis = 1)
        compiledSignal.columns = channels+['Pixels']

        return compiledSignal

    def get_sum_signals(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the sum signals calculated from this analysis task.

        Args:
            fov: the fov to get the sum signals for. If not specified, the
                sum signals for all fovs are returned.

        Returns:
            A pandas data frame containing the sum signal information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_sum_signals(fov) for fov in self.dataSet.get_fovs()]
            )

        return self.dataSet.load_dataframe_from_csv(
            'sequential_signal', self.get_analysis_name(),
            fov, 'signals', index_col=0)

    def _run_analysis(self, fragmentIndex):
        zIndex = int(self.parameters['z_index'])
        channels, geneNames = self.dataSet.get_data_organization()\
            .get_sequential_rounds()

        fovSignal = self._get_sum_signal(fragmentIndex, channels, zIndex)
        normSignal = fovSignal.iloc[:, :-1].div(fovSignal.loc[:, 'Pixels'], 0)
        normSignal.columns = geneNames

        self.dataSet.save_dataframe_to_csv(
                normSignal, 'sequential_signal', self.get_analysis_name(),
                fragmentIndex, 'signals')


class SumSignalMERFISHSpots(SumSignal):

    """
    An analysis task that calculates the signal intensity 
    for each detected spot not used in the codebook
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # probably want to do this for all z index anyways...
        if 'z_indices' not in self.parameters:
            self.parameters['z_indices'] = list(range(len(self.dataSet.get_z_positions())))
        
        # with the decode task we will use the decoded image as a mask
        if 'decode_task' not in self.parameters:
            self.parameters['decode_task'] = None
        else:
            dTask = self.dataSet.load_analysis_task(self.parameters['decode_task'])
            if not dTask.parameters['write_decoded_images']:
                raise analysistask.InvalidParameterException(
                'Invalid decode task: write_decoded_images is False')
            
        # with the segment taks we will also use the segmented image as a mask
        if 'segment_task' not in self.parameters:
            self.parameters['segment_task'] = None

        if 'save_sequential_images' not in self.parameters:
            self.parameters['save_sequential_images'] = False

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']
                ]

    def _save_sequential_images(self, fov: int, images: np.ndarray) -> None:
        num_images = len(images)
        imageDescription = self.dataSet.analysis_tiff_description(
            num_images, 1)
        # using append feature of tifffile - careful
        with self.dataSet.writer_for_analysis_images(
                self, 'sequential', fov, append = True) as outputTif:
            for i in range(num_images):
                outputTif.save(images[i].astype('float32'),
                                photometric='MINISBLACK',
                                contiguous=True,
                                metadata=imageDescription)

    def _extract_signal(self, fov, zIndex, channels, channelImages, 
                        decodedImages = None, cells = None) -> pandas.DataFrame:
        
        # generate a cell mask
        cellMask = np.zeros(self.dataSet.get_image_dimensions(), np.uint16) # will uint16 cv2 work on this?

        if cells is not None:
            cellCoords = []
            for cell in cells:
                regions = cell.get_boundaries()[zIndex]
                if len(regions) == 0:
                    cellCoords.append([])
                else:
                    pixels = []
                    for region in regions:
                        coords = region.exterior.coords.xy
                        xyZip = list(zip(coords[0].tolist(), coords[1].tolist()))
                        pixels.append(np.array(
                                    self.alignTask.global_coordinates_to_fov(
                                        cell.get_fov(), xyZip)))
                    cellCoords.append(pixels)

            cellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))]
            cellIDs_dict = {}
            cellIDs_dict[0] = '0' # default if no cellID?
            for i, (cellID,cellCoord) in enumerate(zip(cellIDs, cellCoords)):
                cellIDs_dict[i+1] = cellID # careful of i+1 here
                cv2.drawContours(cellMask, cellCoord, -1, i+1, -1)
        
        if decodedImages is None:
            decodedImages = np.zeros([3] + self.dataSet.get_image_dimensions(), np.uint16)

        inputImages = np.vstack([cellMask[None, ...], decodedImages, channelImages])
        
        if self.parameters['save_sequential_images']:
            self._save_sequential_images(fov, inputImages)
        
        # swap axes for skimage region props
        inputImages = np.moveaxis(inputImages, 0,-1)

        # we don't want the non decoded areas of -1
        labels = label(decodedImages[0].astype(np.int32), background = -1)
        
        propsDict = regionprops(labels, inputImages)

        # identifier from cell mask
        props_cellmaskid = [np.argmax(np.bincount(
            rp.intensity_image[:,:,0][rp.image].astype(int))) 
            for rp in propsDict]
        # convert identifier back to cell_id
        props_cellid = [cellIDs_dict[k] for k in props_cellmaskid]

        props_bcid = [np.argmax(np.bincount(
            rp.intensity_image[:,:,1][rp.image].astype(int))) 
            for rp in propsDict]
        props_bcmag = [np.mean(rp.intensity_image[:,:,2][rp.image])
            for rp in propsDict]
        props_bcdist = [np.mean(rp.intensity_image[:,:,3][rp.image])
            for rp in propsDict]
        props_filled_area = [rp.filled_area for rp in propsDict]
        props_centroid = np.array([rp.centroid for rp in propsDict])
        props_y = props_centroid[:,0]
        props_x = props_centroid[:,1]

        props_intensities = np.array([np.mean(rp.intensity_image[:,:,4:][rp.image], axis = 0)
            for rp in propsDict]).T

        propsOut = pandas.DataFrame(
            {
                'cellid':props_cellid,
                'barcode_id':props_bcid,
                'mag':props_bcmag,
                'dist':props_bcdist,
                'area':props_filled_area,
                'x':props_x,
                'y':props_y})
        
        propsOut['zIndex'] = zIndex
        for ch, vals in zip(channels, props_intensities):
            # make sure to have unique channel names
            channel_name = self.dataSet.get_data_organization().get_data_channel_name(ch)
            #propsOut[str(ch)] = vals
            propsOut[channel_name] = vals

        return propsOut

    def _get_sum_signal(self, fov, channels, zIndex):

        fTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])

        cells = sTask.get_feature_database().read_features(fov)

        # get decoded images
        dTask = self.dataSet.load_analysis_task(self.parameters['decode_task'])
        decodedImages = np.array([self.dataSet.get_analysis_image(
            dTask.analysisName, 'decoded', fov, 3, zIndex, i) 
            for i in range(3)])

        # these are our sequential images
        imgs = []
        for ch in channels:
            img = fTask.get_aligned_image(fov, ch, zIndex)
            if self.parameters['apply_highpass']:
                highPassSigma = self.parameters['highpass_sigma']
                highPassFilterSize = int(2 * np.ceil(3 * highPassSigma) + 1)
                img = imagefilters.high_pass_filter(img,
                                                    highPassFilterSize,
                                                    highPassSigma)
            imgs.append(img)
        imgs = np.array(imgs)

        compiledSignal = self._extract_signal(fov, zIndex, channels, imgs, 
                                         decodedImages = decodedImages,
                                         cells = cells)
        compiledSignal['fov'] = fov

        return compiledSignal

    def get_sum_signals(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the sum signals calculated from this analysis task.

        Args:
            fov: the fov to get the sum signals for. If not specified, the
                sum signals for all fovs are returned.

        Returns:
            A pandas data frame containing the sum signal information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_sum_signals(fov) for fov in self.dataSet.get_fovs()]
            )

        return self.dataSet.load_dataframe_from_csv(
            'sequential_signal', self.get_analysis_name(),
            fov, 'signals')

    def _run_analysis(self, fragmentIndex):
        
        channels, geneNames = self.dataSet.get_data_organization()\
            .get_sequential_rounds()

        allSignal = pandas.DataFrame()
        for zInd in self.parameters['z_indices']:
            fovSignal = self._get_sum_signal(fragmentIndex, channels, int(zInd))
            allSignal = pandas.concat([allSignal, fovSignal], axis = 0)

        self.dataSet.save_dataframe_to_csv(
                allSignal, 'sequential_signal', self.get_analysis_name(),
                fragmentIndex, 'signals')

class ExportSumSignals(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['sequential_task']]

    def _run_analysis(self):
        sTask = self.dataSet.load_analysis_task(
                    self.parameters['sequential_task'])
        signals = sTask.get_sum_signals()

        self.dataSet.save_dataframe_to_csv(
                    signals, 'sequential_sum_signals',
                    self.get_analysis_name())
