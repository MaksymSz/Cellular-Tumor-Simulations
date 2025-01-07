import matplotlib
import platform

if platform.system() == 'Windows':
    backend = 'TkAgg'
else:
    backend = 'Qt5Agg'
matplotlib.use(backend)

import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)

if sys.version_info[0] == 3:
    from tkinter import *
    from tkinter.ttk import Notebook
else:
    from Tkinter import *
    from ttk import Notebook


class GUI:
    def __init__(self, title='PyCX Simulator', interval=0, stepSize=1, parameterSetters=[]):
        self.titleText = title
        self.timeInterval = interval
        self.stepSize = stepSize
        self.parameterSetters = parameterSetters
        self.varEntries = {}
        self.statusStr = ""

        self.running = False
        self.modelFigure = None
        self.currentStep = 0

        self.rootWindow = Tk()
        self.statusText = StringVar(self.rootWindow, value=self.statusStr)
        self.setStatusStr("Simulation not yet started")

        self.rootWindow.wm_title(self.titleText)
        self.rootWindow.protocol('WM_DELETE_WINDOW', self.quitGUI)
        self.rootWindow.geometry('450x300')
        self.rootWindow.columnconfigure(0, weight=1)
        self.rootWindow.rowconfigure(0, weight=1)

        self.notebook = Notebook(self.rootWindow)
        self.notebook.pack(side=TOP, padx=2, pady=2)

        self.frameRun = Frame(self.rootWindow)
        self.frameSettings = Frame(self.rootWindow)
        self.frameParameters = Frame(self.rootWindow)
        self.frameInformation = Frame(self.rootWindow)

        self.notebook.add(self.frameRun, text="Run")
        self.notebook.pack(expand=NO, fill=BOTH, padx=5, pady=5, side=TOP)

        self.status = Label(self.rootWindow, width=40, height=3, relief=SUNKEN, bd=1, textvariable=self.statusText)
        self.status.pack(side=TOP, fill=X, padx=5, pady=5, expand=NO)

        # frameRun
        self.runPauseString = StringVar(self.rootWindow)
        self.runPauseString.set("Run")
        self.buttonRun = Button(self.frameRun, width=30, height=2, textvariable=self.runPauseString,
                                command=self.runEvent)
        self.buttonRun.pack(side=TOP, padx=5, pady=5)
        self.showHelp(self.buttonRun, "Runs the simulation (or pauses the running simulation)")

        self.buttonStep = Button(self.frameRun, width=30, height=2, text='Step Once', command=self.stepOnce)
        self.buttonStep.pack(side=TOP, padx=5, pady=5)
        self.showHelp(self.buttonStep, "Steps the simulation only once")

        self.buttonReset = Button(self.frameRun, width=30, height=2, text='Reset', command=self.resetModel)
        self.buttonReset.pack(side=TOP, padx=5, pady=5)
        self.showHelp(self.buttonReset, "Resets the simulation")

        self.buttonPlot = Button(self.frameRun, width=30, height=2, text='Plot', command=self.runPlotFunc)
        self.buttonPlot.pack(side=TOP, padx=5, pady=5)
        self.showHelp(self.buttonPlot, "Generates a plot based on the model state")

        # frameSettings
        can = Canvas(self.frameSettings)
        lab = Label(can, width=25, height=1, text="Step size ", justify=LEFT, anchor=W, takefocus=0)
        lab.pack(side='left')

        self.stepScale = Scale(can, from_=1, to=50, resolution=1, command=self.changeStepSize, orient=HORIZONTAL,
                               width=25, length=150)
        self.stepScale.set(self.stepSize)
        self.showHelp(self.stepScale,
                      "Skips model redraw during every [n] simulation steps\nResults in a faster model run.")
        self.stepScale.pack(side='left')

        can.pack(side='top')

        can = Canvas(self.frameSettings)
        lab = Label(can, width=25, height=1, text="Step visualization delay in ms ", justify=LEFT, anchor=W,
                    takefocus=0)
        lab.pack(side='left')
        self.stepDelay = Scale(can, from_=0, to=max(2000, self.timeInterval),
                               resolution=10, command=self.changeStepDelay, orient=HORIZONTAL, width=25, length=150)
        self.stepDelay.set(self.timeInterval)
        self.showHelp(self.stepDelay, "The visualization of each step is delays by the given number of milliseconds.")
        self.stepDelay.pack(side='left')

        can.pack(side='top')

        # frameInformation
        scrollInfo = Scrollbar(self.frameInformation)
        self.textInformation = Text(self.frameInformation, width=45, height=13, bg='lightgray', wrap=WORD,
                                    font=("Courier", 10))
        scrollInfo.pack(side=RIGHT, fill=Y)
        self.textInformation.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollInfo.config(command=self.textInformation.yview)
        self.textInformation.config(yscrollcommand=scrollInfo.set)

        # ParameterSetters
        for variableSetter in self.parameterSetters:
            can = Canvas(self.frameParameters)
            lab = Label(can, width=25, height=1, text=variableSetter.__name__ + " ", anchor=W, takefocus=0)
            lab.pack(side='left')
            ent = Entry(can, width=11)
            ent.insert(0, str(variableSetter()))
            if variableSetter.__doc__ != None and len(variableSetter.__doc__) > 0:
                self.showHelp(ent, variableSetter.__doc__.strip())
            ent.pack(side='left')
            can.pack(side='top')
            self.varEntries[variableSetter] = ent

        if len(self.parameterSetters) > 0:
            self.buttonSaveParameters = Button(self.frameParameters, width=50, height=1,
                                               command=self.saveParametersCmd,
                                               text="Save parameters to the running model", state=DISABLED)
            self.showHelp(self.buttonSaveParameters,
                          "Saves the parameter values.\nNot all values may take effect on a running model\nA model reset might be required.")
            self.buttonSaveParameters.pack(side='top', padx=5, pady=5)
            self.buttonSaveParametersAndReset = Button(self.frameParameters, width=50, height=1,
                                                       command=self.saveParametersAndResetCmd,
                                                       text="Save parameters to the model and reset the model")
            self.showHelp(self.buttonSaveParametersAndReset, "Saves the given parameter values and resets the model")
            self.buttonSaveParametersAndReset.pack(side='top', padx=5, pady=5)

    def setStatusStr(self, newStatus):
        self.statusStr = newStatus
        self.statusText.set(self.statusStr)

    def changeStepSize(self, val):
        self.stepSize = int(val)

    def changeStepDelay(self, val):
        self.timeInterval = int(val)

    def saveParametersCmd(self):
        for variableSetter in self.parameterSetters:
            variableSetter(float(self.varEntries[variableSetter].get()))
            self.setStatusStr("New parameter values have been set")

    def saveParametersAndResetCmd(self):
        self.saveParametersCmd()
        self.resetModel()

    def runEvent(self):
        self.running = not self.running
        if self.running:
            self.rootWindow.after(self.timeInterval, self.stepModel)
            self.runPauseString.set("Pause")
            self.buttonStep.configure(state=DISABLED)
            self.buttonReset.configure(state=DISABLED)
            if len(self.parameterSetters) > 0:
                self.buttonSaveParameters.configure(state=NORMAL)
                self.buttonSaveParametersAndReset.configure(state=DISABLED)
        else:
            self.runPauseString.set("Continue Run")
            self.buttonStep.configure(state=NORMAL)
            self.buttonReset.configure(state=NORMAL)
            if len(self.parameterSetters) > 0:
                self.buttonSaveParameters.configure(state=NORMAL)
                self.buttonSaveParametersAndReset.configure(state=NORMAL)

    def stepModel(self):
        if self.running:
            self.modelStepFunc()
            self.currentStep += 1
            self.setStatusStr("Step " + str(self.currentStep))
            if (self.currentStep) % self.stepSize == 0:
                self.drawModel()
            self.rootWindow.after(int(self.timeInterval * 1.0 / self.stepSize), self.stepModel)

    def stepOnce(self):
        self.running = False
        self.runPauseString.set("Continue Run")
        self.modelStepFunc()
        self.currentStep += 1
        self.setStatusStr("Step " + str(self.currentStep))
        self.drawModel()
        if len(self.parameterSetters) > 0:
            self.buttonSaveParameters.configure(state=NORMAL)

    def resetModel(self):
        self.running = False
        self.runPauseString.set("Run")
        self.modelInitFunc()
        self.currentStep = 0
        self.setStatusStr("Model has been reset")
        self.drawModel()

    def drawModel(self):
        plt.ion()
        if self.modelFigure == None or self.modelFigure.canvas.manager.window == None:
            self.modelFigure = plt.figure()
        self.modelDrawFunc()
        self.modelFigure.canvas.manager.window.update()
        plt.show()

    def runPlotFunc(self):
        if self.modelPlotFunc:
            self.modelPlotFunc()
            self.setStatusStr("Plot function executed")
        else:
            self.setStatusStr("No plot function defined")

    def start(self, func=[]):
        if len(func) == 4:
            self.modelInitFunc = func[0]
            self.modelDrawFunc = func[1]
            self.modelStepFunc = func[2]
            self.modelPlotFunc = func[3]
            if self.modelStepFunc.__doc__:
                self.showHelp(self.buttonStep, self.modelStepFunc.__doc__.strip())
            if self.modelInitFunc.__doc__:
                self.textInformation.config(state=NORMAL)
                self.textInformation.delete(1.0, END)
                self.textInformation.insert(END, self.modelInitFunc.__doc__.strip())
                self.textInformation.config(state=DISABLED)

            self.modelInitFunc()
            self.drawModel()
        self.rootWindow.mainloop()

    def quitGUI(self):
        self.running = False
        self.rootWindow.quit()
        plt.close('all')
        self.rootWindow.destroy()

    def showHelp(self, widget, text):
        def setText(self):
            self.statusText.set(text)
            self.status.configure(foreground='blue')

        def showHelpLeave(self):
            self.statusText.set(self.statusStr)
            self.status.configure(foreground='black')

        widget.bind("<Enter>", lambda e: setText(self))
        widget.bind("<Leave>", lambda e: showHelpLeave(self))
