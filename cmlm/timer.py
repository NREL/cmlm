import time
import numpy as np

class timer:
    
    def __init__(self, task = 'Program', level = 0, verbose=1, start=None):
        # verbose = 0: timings only printed at end.
        #          -1: no printing
        #          >0: printed on the fly for all levels < verbose and at the end
        # start: immediately start timing a task
        self.maintask  = task
        self.tstart     = time.time()
        self.verbose = verbose
        self.tasks   = []
        self.subtimers = {}
        self.ncalls  = 0
        self.cumtime = 0.0
        self.level   = level
        self.running_subtask = False
        self.present_task = None

        # Print a startup message if verbosity dictates it
        if verbose > 0 and self.level is 0:
            print ("Wall start time:" + str(time.ctime()) )

        # Start a subtimer if desired
        if start is not None:
            self.start(start)
            
    def start(self,task):

        if self.running_subtask:
            # Go down a level if there is a task presently running subtasks
            self.subtimers[self.present_task].start(task)

        else :
            # Otherwise, start a task on this level
            
            # Stop the previous task timer if it's running
            if self.present_task is not None:
                self.stop()

            # Create new task timer if needed
            if task not in self.tasks:
                self.tasks.append(task)
                self.subtimers[task] = timer(task = task,
                                             level=self.level+1,
                                             verbose = self.verbose)
                self.present_task = task

            else:
                #restart an existing timer
                self.subtimers[task].tstart = time.time()
                self.present_task = task

            # print some data if verbose
            if self.verbose > self.level:
                print('Starting level '  + str(self.level) + ' task: ' + task, flush=True)

    def start_subtasks(self):
        if self.running_subtask:
            self.subtimers[self.present_task].start_subtasks()

        else :
            self.running_subtask = True

    def stop_subtasks(self):
        if self.present_task is not None:
            if self.subtimers[self.present_task].running_subtask:
                self.subtimers[self.present_task].stop_subtasks()
            else :
                if self.subtimers[self.present_task].present_task is not None:
                    self.subtimers[self.present_task].stop()
                self.running_subtask = False
        else:
            self.running_subtask = False

    def stop(self):

        if self.running_subtask:
            # Go down a level if there is a taskpresently running subtasks
            self.subtimers[self.present_task].stop()
            
        else :
            # stop the running task on this level
            runtime =  time.time() - self.subtimers[self.present_task].tstart
            self.subtimers[self.present_task].ncalls += 1
            self.subtimers[self.present_task].cumtime += runtime
            
            # print some data if verbose
            if self.verbose > self.level:
                print(self.present_task + ' ended after ' + str(runtime) , flush=True)

            self.present_task = None
                
    def finalize(self, totaltime=None):
        # should put in something to stop an existing tasks
        while self.present_task is not None:
            self.stop()
            self.stop_subtasks()
            
        if self.verbose >= 0:
            if totaltime is None :
                self.cumtime = time.time() - self.tstart
                totaltime = self.cumtime
                print ('')
                print (  '{:18s}'.format('Task Name' )
                       + '{:>12s}'.format('Time')
                       + '{:>8s}'.format('%')
                       + '{:>8s}'.format('Ncalls'))
            
            print(''.join(['    '] * self.level)
                  + '{:18s}'.format(self.maintask + ': ' )
                  + '{:12.3e}'.format(self.cumtime)
                  + '{:8.1f}'.format(100*self.cumtime/totaltime)
                  + '{:8d}'.format(self.ncalls ))
            for task in self.tasks:
                self.subtimers[task].finalize(self.cumtime)

class trackerbar:

    def __init__(self, ntotal, name=None, nelems = 50):
        self.ntotal = ntotal
        self.nelems = nelems
        self.current = -1

        if name is None:
            self.name = ''
        else:
            self.name = name + ': '
        
    def update(self, current=None, task=''):
        if current is None:
            self.current+=1
        else:
            self.current = current
            
        n = int((self.current+1)/float(self.ntotal)*self.nelems)
        print(self.name + '[' + '='*n + ' '*(self.nelems-n) + '] '
              + str(task) , end = '\r')
        
    def finalize(self):
        print ('', end = '\r')
        print(self.name+'[' + '='*self.nelems+'] ' + 'done' )
    
# ti = timer(verbose=100)
# ti.start('dogs')
# np.random.rand(100000)

# ti.start_subtasks()
# ti.start('a')
# np.random.rand(100000)
# ti.start('b')
# np.random.rand(100000)
# ti.stop_subtasks()

# ti.start('cats')
# ti.start_subtasks()
# ti.start('a')
# np.random.rand(100000)
# ti.start('b')
# np.random.rand(100000)
# ti.start('b')
# np.random.rand(100000)
# ti.start('b')
# np.random.rand(100000)
# ti.start('a')
# np.random.rand(100000)
# ti.stop_subtasks()
# ti.stop()
# ti.finalize()

# ti = timer(verbose=100)
# ti.start('dogs')
# ti.start('cats')
# ti.stop()
# ti.finalize()
