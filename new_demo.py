import Tkinter as tk
import tkMessageBox
from PIL import ImageTk, Image
import os
import sys
import pickle

import myHRNN


class demo(object):
    def __init__(self):
        pass

    ###
    # Draw initial layout of GUI
    ###
    def setup(self):
        # CREATE LEVEL VARIABLE & INIZIALISE CLASS
        self.root = tk.Tk()
        self.root.bind('<Escape>', self.close)
        # https://stackoverflow.com/questions/35962694/what-is-the-difference-between-the-title-method-and-wm-title-method-in-the-t/35962789
        self.root.wm_title("Image to Paragraph HRNN demo")
        # https://stackoverflow.com/questions/21958534/how-can-i-prevent-a-window-from-being-resized-with-tkinter
        self.root.geometry("1200x900")
        self.root.maxsize(width=1200, height=900)
        self.root.resizable(width=False, height=False)
        
        self.main_panel = tk.Frame(self.root, width=1200, height=800, relief='raised', borderwidth=5, background="bisque")
        self.main_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # DIVIDE MAIN PANEL IN TOP AND BOTTOM PANEL
        self.top_panel = tk.Frame(self.main_panel, width=1190, height=250, relief='raised', borderwidth=5, background="bisque")
        self.top_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.bottom_panel = tk.Frame(self.main_panel, width=1190, height=540, relief='raised', borderwidth=5, background="bisque")
        self.bottom_panel.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # DIVIDE TOP PANEL IN TOP AND BOTTOM
        self.top_panel_top = tk.Frame(self.top_panel, height=275, background="bisque")
        self.top_panel_top.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        #self.top_panel_top.grid(row=1, column=0, columnspan=3, rowspan=5)   
        self.separator_top = tk.Frame(self.top_panel, height=2, bd=1, relief=tk.SUNKEN)
        self.separator_top.pack(fill=tk.BOTH, padx=5, pady=5)
        #self.separator_top.grid(row=5, padx=5, pady=5, rowspan=5)
        self.top_panel_bottom = tk.Frame(self.top_panel, height=275, background="bisque")
        self.top_panel_bottom.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        #self.top_panel_bottom.grid(row=7, column=0, columnspan=3, rowspan=5)

        # DIVIDE BOTTOM PANEL IN LEFT AND RIGHT
        #self.bottom_panel_left = tk.Frame(self.bottom_panel, width=800, background="bisque")
        #self.bottom_panel_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #self.separator_bottom.pack(padx=5, pady=5)
        #self.separator_bottom = tk.Frame(self.bottom_panel, width=1, height=550, bd=1, relief=tk.SUNKEN)
        #self.bottom_panel_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        ##self.bottom_panel_right = tk.Frame(self.bottom_panel, width=400, background="bisque")

        # CREATE DROPDOWN MENU FOR AVAILABLE TRAINED MODELS
        self.models_bias_label = tk.Label(self.top_panel_top, text="Model type:")
        self.models_bias_label.grid(row=1, column=1)
        self.models_bias = tk.StringVar()
        self.models_bias.set('no_bias')
        self.models_bias_dropdown = tk.OptionMenu(self.top_panel_top, self.models_bias,'no_bias', 'normalized_bias', 'bias')
        self.models_bias_dropdown.grid(row=2, column=1, sticky="ew")
        self.models_bias.trace('w', self.changeNOfEpochs)
        
        # https://stackoverflow.com/questions/45441885/python-tkinter-creating-a-dropdown-select-bar-from-a-list
        self.models_label = tk.Label(self.top_panel_top, text="Number of epochs:")
        self.models_label.grid(row=1, column=2)
        self.models = tk.StringVar()
        self.models.set('epoch-0')
        self.models_dropdown = tk.OptionMenu(self.top_panel_top, self.models,'epoch-0', 'epoch-5','epoch-10', 'epoch-15', 'epoch-20', 'epoch-25', 'epoch-30', 'epoch-35', 'epoch-40', 'epoch-45', 'epoch-50', 'epoch-55', 'epoch-60','epoch-65','epoch-70', 'epoch-75', 'epoch-80', 'epoch-85', 'epoch-90')
        self.models_dropdown.grid(row=2, column=2)

        # CREATE BUTTON TO LOAD THE DESIRED MODEL
        self.load_model_button = tk.Button(master=self.top_panel_top, height=1, text='Load Model', command=self.load_model, justify='center', background="bisque", relief=tk.RAISED)
        self.load_model_button.grid(row=2, column=4)
        
        # CREATE BUTTON TO SHOW GRAPHS
        self.show_graphs_button = tk.Button(master=self.top_panel_top, height=1, text='Show Graphs', command=self.show_graphs, justify='center', background="bisque", relief=tk.RAISED)
        self.show_graphs_button.grid(row=2, column=6)

        # CREATE BUTTON TO UNLOAD THE ACTIVE MODEL
        self.unload_model_button = tk.Button(master=self.top_panel_top, height=1, text='Unload Model', command=self.unload_model, justify='center', background="bisque", relief=tk.RAISED)
        self.unload_model_button.grid(row=2, column=8)

        #
        self.top_panel_top.grid_rowconfigure(0, weight=1)
        self.top_panel_top.grid_rowconfigure(1, weight=1)
        self.top_panel_top.grid_rowconfigure(3, weight=1)

        self.top_panel_top.grid_columnconfigure(0, weight=1)
        self.top_panel_top.grid_columnconfigure(3, weight=1)
        self.top_panel_top.grid_columnconfigure(5, weight=1)
        self.top_panel_top.grid_columnconfigure(7, weight=1)
        self.top_panel_top.grid_columnconfigure(9, weight=1)

        # CREATE ENTRY BOX FOR IMAGE ID TO BE TESTED
        # http://effbot.org/tkinterbook/entry.htm
        self.image_path_entry = tk.Entry(master=self.top_panel_bottom, text='Image path', justify='center')
        # https://stackoverflow.com/questions/41769497/python-tkinter-what-are-the-correct-values-for-the-anchor-option-in-the-message
        self.image_path_entry.grid(row=3, column=2, sticky="ew")

        # CREATE THE BUTTON TO FETCH THE DESIRED IMAGE ID AND TRIGGER THE TESTING
        self.image_path_button = tk.Button(master=self.top_panel_bottom, text='Test', command=self.test_image, justify='center', background="bisque", relief=tk.RAISED)
        self.image_path_button.grid(row=3, column=3, sticky="ew")

        self.top_panel_bottom.grid_rowconfigure(0, weight=1)
        self.top_panel_bottom.grid_rowconfigure(1, weight=1)
        self.top_panel_bottom.grid_rowconfigure(3, weight=1)
        self.top_panel_bottom.grid_rowconfigure(4, weight=1)
        self.top_panel_bottom.grid_rowconfigure(5, weight=1)

        self.top_panel_bottom.grid_columnconfigure(0, weight=1)
        self.top_panel_bottom.grid_columnconfigure(1, weight=1)
        self.top_panel_bottom.grid_columnconfigure(4, weight=1)
        self.top_panel_bottom.grid_columnconfigure(5, weight=1)
        self.top_panel_bottom.grid_columnconfigure(6, weight=1)

        # CREATE PANEL TO DISPLAY TESTED IMAGE
        self.image_panel = tk.Label(master=self.bottom_panel, image="", justify='center', background="bisque")
        #self.image_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_panel.grid(row=2, column=1, columnspan=3, sticky="nsew")

        # CREATE PANEL TO DISPLAY PARAGRAPH RETURNED FROM TESTING
        self.paragraph_panel = tk.Label(master=self.bottom_panel, text="", justify='center', background="bisque")
        #self.paragraph_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.paragraph_panel.grid(row=2, column=5, columnspan=2, sticky="nsew")
        
        self.bottom_panel.grid_rowconfigure(0, weight=1)
        self.bottom_panel.grid_rowconfigure(1, weight=1)
        self.bottom_panel.grid_rowconfigure(3, weight=1)

        self.bottom_panel.grid_columnconfigure(0, weight=1)
        self.bottom_panel.grid_columnconfigure(4, weight=1)
        self.bottom_panel.grid_columnconfigure(7, weight=1)
        

        # https://stackoverflow.com/questions/44826267/setting-tk-frame-width-and-height
        for frame in [self.main_panel, self.top_panel, self.bottom_panel]:
            frame.pack(expand=True, fill='both')
            frame.pack_propagate(0)

        for widget in [self.image_panel, self.paragraph_panel]:
            widget.pack(expand=True, fill='x', anchor='s')
        
        for widget in [self.models_dropdown, self.image_path_entry, self.unload_model_button, self.load_model_button, self.image_path_button]:
            widget.grid(sticky='wse')
    
    

    ###
    # Open session for selected model
    ###
    def load_model(self):
        
        self.global_session, self.model_features, self.model_paragraph, self.model_probabilities, self.model_vector_sentenceTopic = myHRNN.loadModel(self.models_bias.get(), self.models.get())

    ####
    # Test image on loaded model
    ####
    def test_image(self):
        # get image name
        # https://stackoverflow.com/questions/47378715/tkinter-how-to-get-the-value-of-an-entry-widget
        self.image_name_entry_string = self.image_path_entry.get()
        
        # find correct image path
        self.testing_path = "/Users/costanzamariaimprota/dataset/testing/" + self.image_name_entry_string + ".jpg"
        self.training_path = "/Users/costanzamariaimprota/dataset/training/" + self.image_name_entry_string + ".jpg"
        self.validation_path = "/Users/costanzamariaimprota/dataset/validation/" + self.image_name_entry_string + ".jpg"
        self.personal_path = "/Users/costanzamariaimprota/dataset/personal/" + self.image_name_entry_string + ".jpg"

        if os.path.exists(self.testing_path):
            self.show_image(self.testing_path)
        elif os.path.exists(self.training_path):
            self.show_image(self.training_path)
        elif os.path.exists(self.validation_path):
            self.show_image(self.validation_path)
        elif os.path.exists(self.personal_path):
            self.show_image(self.personal_path)
        else:
            # invalid image id
            #https://pythonspot.com/tk-message-box/
            tkMessageBox.showinfo("Error!", "The image cannot be found!")
            # https://stackoverflow.com/questions/6190776/what-is-the-best-way-to-exit-a-function-which-has-no-return-value-in-python-be
            return
        
        # find features path
        self.path_personalFeatures = "/Users/costanzamariaimprota/ThirdYearProject/densecap_features/personal/" + self.image_name_entry_string + ".h5"
        self.path_trainingFeatures = "/Users/costanzamariaimprota/ThirdYearProject/densecap_features/training/" + self.image_name_entry_string + ".h5"
        self.path_testingFeatures = "/Users/costanzamariaimprota/ThirdYearProject/densecap_features/testing/" + self.image_name_entry_string + ".h5"
        self.path_validationFeatures = "/Users/costanzamariaimprota/ThirdYearProject/densecap_features/validation/" + self.image_name_entry_string + ".h5"
        
        if os.path.exists(self.path_personalFeatures):
            self.get_loaded_model_paragraph(self.path_personalFeatures)
        elif os.path.exists(self.path_trainingFeatures):
            self.get_loaded_model_paragraph(self.path_trainingFeatures)
        elif os.path.exists(self.path_testingFeatures):
            self.get_loaded_model_paragraph(self.path_testingFeatures)
        elif os.path.exists(self.path_validationFeatures):
            self.get_loaded_model_paragraph(self.path_validationFeatures)
        else:
            # invalid image id
            #https://pythonspot.com/tk-message-box/
            tkMessageBox.showinfo("Error!", "There is no features file for this image! Did you run densecap?")
            # https://stackoverflow.com/questions/6190776/what-is-the-best-way-to-exit-a-function-which-has-no-return-value-in-python-be
            return

    ################################
    ### Display selected image
    #
    def show_image(self, image_path):
        self.original = Image.open(image_path)
        resized = self.original.resize((590, 600),Image.ANTIALIAS)
        self.image = ImageTk.PhotoImage(resized)
         # https://stackoverflow.com/questions/610883/how-to-know-if-an-object-has-an-attribute-in-python 
        if hasattr(self, 'image_panel'):
            # clear panel containing image
            # https://stackoverflow.com/questions/33302779/python-tkinter-remove-delete-image-from-label
            self.image_panel.config(image="")
            self.image_panel.pack_forget()
        self.image_panel = tk.Label(self.bottom_panel, image = self.image)
        #self.image_panel.pack()
        self.image_panel.grid(row=2, column=1, columnspan=3)
        self.image_panel.grid_propagate(0)

    ################################
    ### Retrieve paragraph for image tested on loaded model
    #
    def get_loaded_model_paragraph(self, features_path):
        self.paragraph = myHRNN.singleTesting(features_path, self.image_name_entry_string, self.global_session, self.model_features, self.model_paragraph, self.model_probabilities, self.model_vector_sentenceTopic)
        if hasattr(self, 'paragraph_panel'):
            # clear panel containing paragraph
            # https://stackoverflow.com/questions/33302779/python-tkinter-remove-delete-image-from-label
            self.paragraph_panel.config(text="")
            self.paragraph_panel.pack_forget()
        if "training" in features_path:
            paragraphs = pickle.load(open('./dataset/training_groundtruth', 'rb'))
            textParagraph = paragraphs[int(self.image_name_entry_string)][1]
            if '' in textParagraph:
                textParagraph.remove('')
            if ' ' in textParagraph:
                textParagraph.remove(' ')
            self.paragraph += '\n \n Ground truth: \n'
            self.paragraph += '. '.join(textParagraph)
        self.paragraph_panel = tk.Message(master=self.bottom_panel, text=self.paragraph, justify=tk.LEFT)
        #self.paragraph_panel.pack()
        self.paragraph_panel.grid(row=2, column=5, columnspan=2, sticky="nsew")
    
    #https://stackoverflow.com/questions/35924690/tkinter-image-wont-show-up-in-new-window
    def show_graphs(self):
        '''
        nOfEpochs = self.models.get().split("-")[1]
        self.show_loss(nOfEpochs)
        self.show_lossSentence(nOfEpochs)
        self.show_lossWord(nOfEpochs)
        '''
        self.nOfEpochs = self.models.get().split("-")[1]
        self.newWindow = tk.Toplevel()
        self.newWindow.title("Graphs of Loss for current model")
        
        self.path_img = "./"  + self.models_bias.get() +"_graphs/" + self.nOfEpochs + ".png"
        self.path_imgSentence = "./"  + self.models_bias.get() +"_graphs/" + self.nOfEpochs + "-sentence.png"
        self.path_imgWord = "./"  + self.models_bias.get() +"_graphs/" + self.nOfEpochs + "-word.png"

        if(os.path.exists(self.path_img) and os.path.exists(self.path_imgSentence) and os.path.exists(self.path_imgSentence)):
            self.img = Image.open(self.path_img)
            self.imgSentence = Image.open(self.path_imgSentence)
            self.imgWord = Image.open(self.path_imgWord)

            self.img_resized = self.img.resize((550, 450),Image.ANTIALIAS)
            self.imgSentence_resized = self.imgSentence.resize((550, 450),Image.ANTIALIAS)
            self.imgWord_resized = self.imgWord.resize((550, 450),Image.ANTIALIAS)

            self.img = ImageTk.PhotoImage(self.img_resized)
            self.imgSentence = ImageTk.PhotoImage(self.imgSentence_resized)
            self.imgWord = ImageTk.PhotoImage(self.imgWord_resized)

            self.new_panel = tk.Frame(self.newWindow)

            self.img_label = tk.Label(self.new_panel, text="HRNN Loss")
            self.img_label.grid(row=1, column=1, columnspan=1)
            self.imgSentence_label = tk.Label(self.new_panel, text="Sentence RNN Loss")
            self.imgSentence_label.grid(row=1, column=4, columnspan=1)
            self.imgWord_label = tk.Label(self.new_panel, text="Word RNN Loss")
            self.imgWord_label.grid(row=1, column=7, columnspan=1)

            self.img_panel = tk.Label(self.new_panel, image = self.img)
            self.img_panel.grid(row=2, column=1, columnspan=1)
            self.imgSentence_panel = tk.Label(self.new_panel, image = self.imgSentence)
            self.imgSentence_panel.grid(row=2, column=4, columnspan=1)
            self.imgWord_panel = tk.Label(self.new_panel, image = self.imgWord)
            self.imgWord_panel.grid(row=2, column=7, columnspan=1)

            self.new_panel.grid_rowconfigure(0, weight=1)
            self.new_panel.grid_rowconfigure(3, weight=1)

            self.new_panel.grid_columnconfigure(3, weight=1)
            self.new_panel.grid_columnconfigure(6, weight=1)
            self.new_panel.grid_columnconfigure(9, weight=1)
        
            self.new_panel.pack()
        else:
            tkMessageBox.showinfo("Error!", "The graphs cannot be found!")
            # https://stackoverflow.com/questions/6190776/what-is-the-best-way-to-exit-a-function-which-has-no-return-value-in-python-be
            return
            

        
        
        #self.image_panel.grid_propagate(0)    
    ################################
    ### Close active Tensorflow session 
    #
    def unload_model(self):
        myHRNN.unloadModel(self.global_session)
    
    ################################
    ### Run GUI
    #
    def mainloop(self):
        tk.mainloop()
    
    def close(self, event=None):
        self.root.destroy()

    # https://mail.python.org/pipermail/tutor/2005-July/040018.html
    def changeNOfEpochs(*args):
        demo.models_dropdown.destroy()
        
        if(demo.models_bias.get() == 'no_bias'):
            demo.models_dropdown = tk.OptionMenu(demo.top_panel_top, demo.models,'epoch-0', 'epoch-5','epoch-10', 'epoch-15', 'epoch-20', 'epoch-25', 'epoch-30', 'epoch-35', 'epoch-40', 'epoch-45', 'epoch-50', 'epoch-55', 'epoch-60','epoch-65','epoch-70', 'epoch-75', 'epoch-80', 'epoch-85', 'epoch-90')
        elif (demo.models_bias.get() == 'normalized_bias'):
            demo.models_dropdown = tk.OptionMenu(demo.top_panel_top, demo.models,'epoch-0', 'epoch-5','epoch-10', 'epoch-15', 'epoch-20', 'epoch-25', 'epoch-30', 'epoch-35', 'epoch-40', 'epoch-45', 'epoch-50', 'epoch-55', 'epoch-60','epoch-65','epoch-70', 'epoch-75', 'epoch-80', 'epoch-85', 'epoch-90', 'epoch-95', 'epoch-100', 'epoch-105', 'epoch-110', 'epoch-115', 'epoch-120', 'epoch-125', 'epoch-130')
        else:
            demo.models_dropdown = tk.OptionMenu(demo.top_panel_top, demo.models,'epoch-0', 'epoch-5','epoch-10', 'epoch-15', 'epoch-20', 'epoch-25', 'epoch-30', 'epoch-35', 'epoch-40', 'epoch-45', 'epoch-50', 'epoch-55', 'epoch-60','epoch-65','epoch-70', 'epoch-75', 'epoch-80', 'epoch-85', 'epoch-90', 'epoch-95', 'epoch-100', 'epoch-105', 'epoch-110', 'epoch-115', 'epoch-120')
        demo.models_dropdown.grid(row=2, column=2)

################################
### Create, initialise and run GUI
#
demo = demo()
demo.setup()
demo.mainloop()

