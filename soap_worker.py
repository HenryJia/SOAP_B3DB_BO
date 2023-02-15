import multiprocessing as mp
from multiprocessing import Process, Queue, Value, Lock

# We are in essence creating a pool of workers to calculate the SOAP vectors for a dataset of molecules.
# Now I am aware that multiprocessing.Pool exists, but this is much more flexible.
# I'd rather make my own pool than try to shoehorn the multiprocessing.Pool into my code.

class SOAPWorker(Process):
    '''
    A class to calculate the SOAP vectors for a dataset of molecules in parallel.
    This class is intended to be used with the SOAPPool class.
    
    Parameters
    ----------
    dataset : MoleculeDataset
        The dataset of molecules to be processed.
    parameter_strings : list of str
        A list of strings containing the parameters for the SOAP descriptor.
    '''
    def __init__(self, idx, dataset, parameter_strings, completion_lock):
        self.idx = idx
        self.dataset = dataset
        self.parameter_strings = parameter_strings
        self.completion_lock = completion_lock

    '''
    Calculate the SOAP vectors for the molecules in the dataset.

    Parameters
    ----------
    soaps : Empty list
        The list to store the SOAP vectors in.
    '''
    def run(self, soaps_queue, num_active_workers):
        self.completion_lock.acquire(block=True)
        num_active_workers.value = num_active_workers+1
        self.completion_lock.release()

        soaps = []
        for row in self.dataset.df.itertuples():
            soap = []
            for ps in self.parameter_strings:
                soap += list(descriptors.Descriptor(ps).calc(row.Mol)['data'][0])

            soap = np.array(soap)
            if np.isnan(soap).any():
                warnings.warn("NaN detected in molecule:\n{}".format(row))

            soaps.append(soap)

        soaps_queue.put((idx, soaps))

        # This tells the pool that this worker is done.
        self.completion_lock.acquire(block=True)
        num_active_workers.value = num_active_workers-1
        self.completion_lock.release()

        

class SOAPPool:
    '''
    A class to calculate the SOAP vectors for a dataset of molecules in parallel.
    This class is intended to be used with the SOAPWorker class.
    
    Parameters
    ----------
    '''
    def __init__(self, num_workers=mp.cpu_count()):
        self.num_workers = num_workers

        self.workers = [] # This is a queue of workers that are currently active.
        self.num_active_workers = Value('i', 0)
        self.completion_lock = Lock()

        self.results = []

    def add(self, dataset, parameter_strings):
        soaps = List
        self.worker_queue.put(SOAPWorker(dataset, parameter_strings, self.completion_lock).run(soaps, self.num_active_workers))

    def run(self):
        num_active_workers = 0
        for worker in workers:
            # This is a little bit of a hack. But it works, it's simple and it's not too slow.
            time.sleep(0.1)
            if self.num_active_workers < self.num_workers:
                worker = self.worker_queue.get()
                worker.start()

        for worker in workers:
            worker.join()

        for worker in workers:
            soaps += worker.soaps

        return soaps


        