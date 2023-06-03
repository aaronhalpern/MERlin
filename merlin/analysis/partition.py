import pandas
import numpy as np

from merlin.core import analysistask
from merlin.util import spatialfeature

class PartitionBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        if 'boundary_correction_buffer_size' not in self.parameters:
            self.parameters['boundary_correction_buffer_size'] = 0.5

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['filter_task'],
                self.parameters['assignment_task'],
                self.parameters['alignment_task']]

    def get_partitioned_barcodes(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_partitioned_barcodes(fov)
                 for fov in self.dataSet.get_fovs()]
            )

        return self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov, index_col=0)

    def _run_analysis(self, fragmentIndex):
        if self.parameters['boundary_correction_buffer_size'] == 0:
            self._run_analysis_traditional(fragmentIndex)
        else:
            self._run_analysis_boundary_correction(fragmentIndex)

    def _run_analysis_boundary_correction(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()
        for fi in fovIntersections:
            partialBC = bcDB.get_barcodes(fi)
            if fi == fovIntersections[0]:
                currentFOVBarcodes = partialBC.copy(deep=True)
            else:
                currentFOVBarcodes = pandas.concat(
                    [currentFOVBarcodes, partialBC], 0)

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        print(f'FOV {fragmentIndex} has {len(currentCells)} cells.')

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        countsDF_shrinked = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        # Make a copy of the currentCells with the changed boundaries
        buffer_size = self.parameters['boundary_correction_buffer_size']
        currentCells_expanded = [c.make_a_buffered_copy(buffer_size) for c in currentCells]
        currentCells_shrinked = [c.make_a_buffered_copy(-buffer_size) for c in currentCells]

        # Store the host cells of each barcode
        N_barcodes = currentFOVBarcodes.shape[0]
        host_cells_normal = [[] for i in range(N_barcodes)]
        host_cells_shrinked = [[] for i in range(N_barcodes)]
        host_cells_expanded = [[] for i in range(N_barcodes)]
    
        # Get the host cells for each barcode
        for i in range(len(currentCells)):
            cell_normal = currentCells[i]
            cell_shrinked = currentCells_shrinked[i]
            cell_expanded = currentCells_expanded[i]

            contained_normal = cell_normal.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            
            contained_shrinked = cell_shrinked.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            
            contained_expanded = cell_expanded.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
           
            ids_normal = np.where(contained_normal)[0]
            ids_shrinked = np.where(contained_shrinked)[0]
            ids_expanded = np.where(contained_expanded)[0]

            for j in ids_normal:
                host_cells_normal[j].append(i)
            for j in ids_shrinked:
                host_cells_shrinked[j].append(i)
            for j in ids_expanded:
                host_cells_expanded[j].append(i)

            # Generate a counting matrix for the shrinked cells
            count_shrinked = currentFOVBarcodes[contained_shrinked].groupby('barcode_id').size()
            count_shrinked = count_shrinked.reindex(range(barcodeCount), fill_value=0)
            countsDF_shrinked.loc[cell_shrinked.get_feature_id(), :] = count_shrinked.values.tolist()

        # Assign each barcode to a cell
        barcode_types = list(currentFOVBarcodes['barcode_id'])

        for i in range(N_barcodes):
            # Continue if this barcode is not within any cell
            if len(host_cells_expanded[i]) == 0:
                continue

            barcode_id = barcode_types[i]
            
            # If the barcode is within the shrinked region of a cell, add it to the cell
            if len(host_cells_shrinked[i]) > 0:
                countsDF.iloc[host_cells_shrinked[i][0], barcode_id] += 1
                continue

            # If a cell contains a barcode in its expanded regions but
            # doesn't have it in the interior filter it out
            hce_filtered = [c for c in host_cells_expanded[i] 
                              if countsDF_shrinked.iloc[c, barcode_id] > 0]

            # If the barcode is within the normal region of a cell
            if len(host_cells_normal[i]) > 0:
                cell_normal_id = host_cells_normal[i][0]

                # Assign the barcode to the normal cell if the cell already 
                # have a same type barcode in its interior
                if countsDF_shrinked.iloc[cell_normal_id, barcode_id] > 0:
                    countsDF.iloc[cell_normal_id, barcode_id] += 1
                
                # If any other cell that contains the barcode in the expanded region
                # and have the same barcode type in its interior, assign the barcode 
                # to that cell
                elif len(hce_filtered) > 0:
                    hce = np.random.choice(hce_filtered)
                    countsDF.iloc[hce, barcode_id] += 1

                # If no other filtered cells contains the barcode, asign it to the normal cell
                else:
                    countsDF.iloc[cell_normal_id, barcode_id] += 1
                
                continue

            # If a cell only exist in the expanded region, assign it
            if len(hce_filtered) > 0:
                hce = np.random.choice(hce_filtered)
                countsDF.iloc[hce, barcode_id] += 1

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)
    
    def _run_analysis_traditional(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()
        for fi in fovIntersections:
            partialBC = bcDB.get_barcodes(fi)
            if fi == fovIntersections[0]:
                currentFOVBarcodes = partialBC.copy(deep=True)
            else:
                currentFOVBarcodes = pandas.concat(
                    [currentFOVBarcodes, partialBC], 0)

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = pandas.DataFrame(
            data=np.zeros((len(currentCells), barcodeCount)),
            columns=range(barcodeCount),
            index=[x.get_feature_id() for x in currentCells])

        for cell in currentCells:
            contained = cell.contains_positions(currentFOVBarcodes.loc[:,
                                                ['global_x', 'global_y',
                                                 'z']].values)
            count = currentFOVBarcodes[contained].groupby('barcode_id').size()
            count = count.reindex(range(barcodeCount), fill_value=0)
            countsDF.loc[cell.get_feature_id(), :] = count.values.tolist()

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)


class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['partition_task']]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(
                    self.parameters['partition_task'])
        parsedBarcodes = pTask.get_partitioned_barcodes()

        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'barcodes_per_feature',
                    self.get_analysis_name())
