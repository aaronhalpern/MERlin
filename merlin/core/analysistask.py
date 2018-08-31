import os
import copy
from abc import ABC, abstractmethod
import time
import threading

import numpy as np


class AnalysisAlreadyStartedException(Exception):
    pass


class AnalysisTask(ABC):

    '''
    An abstract class for performing analysis on a DataSet. Subclasses
    should implement the analysis to perform in the run_analysis() function.
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        '''Creates an AnalysisTask object that performs analysis on the
        specified DataSet.

        Args:
            dataSet: the DataSet to run analysis on.
            parameters: a dictionary containing parameters used to run the
                analysis.
            analysisName: specifies a unique identifier for this
                AnalysisTask. If analysisName is not set, the analysis name
                will default to the name of the class.
        '''
        self.dataSet = dataSet
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = copy.deepcopy(parameters)

        if analysisName is None:
            self.analysisName = type(self).__name__
        else:
            self.analysisName = analysisName

        self.parameters['module'] = type(self).__module__
        self.parameters['class'] = type(self).__name__

    def save(self):
        '''Save a copy of this AnalysisTask into the data set.'''
        self.dataSet.save_analysis_task(self)

    def run(self):
        '''Run this AnalysisTask.
        
        Upon completion of the analysis, this function informs the DataSet
        that analysis is complete.
        '''

        logger = self.dataSet.get_logger(self)
        logger.info('Beginning ' + self.get_analysis_name())

        try:
            if self.is_complete() or self.is_running():
                raise AnalysisAlreadyStartedException

            self.dataSet.record_analysis_started(self)
            self._indicate_running()
            self.run_analysis()
            self.dataSet.record_analysis_complete(self)
            logger.info('Completed ' + self.get_analysis_name())
        except Exception as e:
            logger.exception(e)
            self.dataSet.record_analysis_error(self)


    def _indicate_running(self):
        '''A loop that regularly signals to the dataset that this analysis
        task is still running successfully. 

        Once this function is called, the dataset will be notified every 
        minute that this analysis is still running until the analysis
        completes.
        '''
        if self.is_complete() or self.is_error():
            return

        self.dataSet.record_analysis_running(self)
        threading.Timer(30, self._indicate_running).start()

    @abstractmethod
    def run_analysis(self):
        '''Perform the analysis for this AnalysisTask.

        This function should be implemented in all subclasses with the
        logic to complete the analysis.
        '''
        pass

    @abstractmethod
    def get_estimated_memory(self):
        '''Get an estimate of how much memory is required for this
        AnalysisTask.

        Returns:
            a memory estimate in megabytes.
        '''
        pass

    @abstractmethod
    def get_estimated_time(self):
        '''Get an estimate for the amount of time required to complete
        this AnalysisTask.

        Returns:
            a time estimate in minutes.
        '''
        pass

    @abstractmethod
    def get_dependencies(self):
        '''Get the analysis tasks that must be completed before this 
        analysis task can proceed.

        Returns:
            a list containing the names of the analysis tasks that 
                this analysis task depends on
        '''
        pass

    def get_parameters(self):
        '''Get the parameters for this analysis task.

        Returns:
            the parameter dictionary
        '''
        return self.parameters

    def is_error(self):
        '''Determines if an error has occured while running this analysis
        
        Returns:
            True if the analysis is complete and otherwise False.
        '''
        return self.dataSet.check_analysis_error(self)

    def is_complete(self):
        '''Determines if this analysis has completed successfully
        
        Returns:
            True if the analysis is complete and otherwise False.
        '''
        return self.dataSet.check_analysis_done(self)

    def is_running(self):
        '''Determines if this analysis has started.
        
        Returns:
            True if the analysis is complete and otherwise False.
        '''
        return self.dataSet.check_analysis_started(self) and not \
                self.is_complete()

    def is_idle(self):
        '''Determines if this analysis task is expected to be running,
        but has stopped for some reason.
        '''
        if not self.is_running():
            return False

        return self.dataSet.is_analysis_idle(self)


    def get_analysis_name(self):
        '''Get the name for this AnalysisTask.

        Returns:
            the name of this AnalysisTask
        '''
        return self.analysisName


class ParallelAnalysisTask(AnalysisTask):

    '''
    An abstract class for analysis that can be run in multiple parts 
    independently. Subclasses should implement the analysis to perform in 
    the run_analysis() function
    '''

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    @abstractmethod
    def fragment_count(self):
        pass

    def run(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                self.run(i)
        else:
            logger = self.dataSet.get_logger(self, fragmentIndex)
            logger.info('Beginning ' + self.get_analysis_name())
            try: 
                if self.is_complete(fragmentIndex) \
                        or self.is_running(fragmentIndex):
                    raise AnalysisAlreadyStartedException    
                self.dataSet.record_analysis_started(self, fragmentIndex)
                self._indicate_running(fragmentIndex)
                self.run_analysis(fragmentIndex)
                self.dataSet.record_analysis_complete(self, fragmentIndex) 
                logger.info('Completed ' + self.get_analysis_name())
            except Exception as e:
                logger.exception(e)
                self.dataSet.record_analysis_error(self, fragmentIndex)


    def _indicate_running(self, fragmentIndex):
        '''A loop that regularly signals to the dataset that this analysis
        task is still running successfully. 

        Once this function is called, the dataset will be notified every 
        minute that this analysis is still running until the analysis
        completes.
        '''
        if self.is_complete(fragmentIndex) or self.is_error(fragmentIndex):
            return

        self.dataSet.record_analysis_running(self, fragmentIndex)
        threading.Timer(30, self._indicate_running, [fragmentIndex]).start()

    @abstractmethod
    def run_analysis(self, fragmentIndex):
        pass

    def is_error(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if self.is_error(i):
                    return True 

            return False

        else:
            return self.dataSet.check_analysis_error(self, fragmentIndex)

    def is_complete(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if not self.is_complete(i):
                    return False

            return True

        else:
            return self.dataSet.check_analysis_done(self, fragmentIndex)

    def is_running(self, fragmentIndex=None):
        if fragmentIndex is None:
            for i in range(self.fragment_count()):
                if self.is_running(i):
                    return True 

            return False

        else:
            return self.dataSet.check_analysis_started(self, fragmentIndex) \
                    and not self.is_complete(fragmentIndex)

    def is_idle(self, fragmentIndex=None):
        if not self.is_running(fragmentIndex):
            return False

        return self.dataSet.is_analysis_idle(self, fragmentIndex)