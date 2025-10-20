"""Tools to profile and track elapsed time in python code."""

import time


class Timer:

    def __init__(self, task="Program", level=0, verbose=1, start=None):
        # verbose = 0: timings only printed at end.
        #          -1: no printing
        #          >0: printed on the fly for all levels < verbose and at the end
        # start: immediately start timing a task
        self.maintask = task
        self.tstart = time.time()
        self.verbose = verbose
        self.tasks = []
        self.subtimers = {}
        self.ncalls = 0
        self.cumtime = 0.0
        self.level = level
        self.running_subtask = False
        self.present_task = None

        # Print a startup message if verbosity dictates it
        if verbose > 0 and self.level == 0:
            print("Wall start time:" + str(time.ctime()))

        # Start a subtimer if desired
        if start is not None:
            self.start(start)

    def start(self, task):

        if self.running_subtask:
            # Go down a level if there is a task presently running subtasks
            self.subtimers[self.present_task].start(task)

        else:
            # Otherwise, start a task on this level

            # Stop the previous task timer if it's running
            if self.present_task is not None:
                self.stop()

            # Create new task timer if needed
            if task not in self.tasks:
                self.tasks.append(task)
                self.subtimers[task] = Timer(
                    task=task, level=self.level + 1, verbose=self.verbose
                )
                self.present_task = task

            else:
                # restart an existing timer
                self.subtimers[task].tstart = time.time()
                self.present_task = task

            # print some data if verbose
            if self.verbose > self.level:
                print(
                    "Starting level " + str(self.level) + " task: " + task, flush=True
                )

    def start_subtasks(self):
        if self.running_subtask:
            self.subtimers[self.present_task].start_subtasks()

        else:
            self.running_subtask = True

    def stop_subtasks(self):
        if self.present_task is not None:
            if self.subtimers[self.present_task].running_subtask:
                self.subtimers[self.present_task].stop_subtasks()
            else:
                if self.subtimers[self.present_task].present_task is not None:
                    self.subtimers[self.present_task].stop()
                self.running_subtask = False
        else:
            self.running_subtask = False

    def stop(self):

        if self.running_subtask:
            # Go down a level if there is a taskpresently running subtasks
            self.subtimers[self.present_task].stop()

        else:
            # stop the running task on this level
            runtime = time.time() - self.subtimers[self.present_task].tstart
            self.subtimers[self.present_task].ncalls += 1
            self.subtimers[self.present_task].cumtime += runtime

            # print some data if verbose
            if self.verbose > self.level:
                print(self.present_task + " ended after " + str(runtime), flush=True)

            self.present_task = None

    def finalize(self, totaltime=None):
        # should put in something to stop an existing tasks
        while self.present_task is not None:
            self.stop()
            self.stop_subtasks()

        if self.verbose >= 0:
            if totaltime is None:
                self.cumtime = time.time() - self.tstart
                totaltime = self.cumtime
                print("")
                print(
                    f"{'Task Name':18s}" f"{'Time':>12s}" f"{'%':>8s}" f"{'Ncalls':>8s}"
                )

            print(
                "".join(["    "] * self.level) + f"{self.maintask + ': ':18s}"
                f"{self.cumtime:12.3e}"
                f"{100 * self.cumtime / totaltime:8.1f}"
                f"{self.ncalls:8d}"
            )
            for task in self.tasks:
                self.subtimers[task].finalize(self.cumtime)


class TrackerBar:

    def __init__(self, ntotal, name=None, nelems=50):
        self.ntotal = ntotal
        self.nelems = nelems
        self.current = -1

        if name is None:
            self.name = ""
        else:
            self.name = name + ": "

    def update(self, current=None, task=""):
        if current is None:
            self.current += 1
        else:
            self.current = current

        n = int((self.current + 1) / float(self.ntotal) * self.nelems)
        print(
            self.name + "[" + "=" * n + " " * (self.nelems - n) + "] " + str(task),
            end="\r",
        )

    def finalize(self):
        print("", end="\r")
        print(self.name + "[" + "=" * self.nelems + "] " + "done")
