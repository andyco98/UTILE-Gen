#!/usr/bin/env python
# coding: utf-8

# In[1]:

from PIL import Image
import os
import cv2
from statistics import mean
import imagej
ij = imagej.init(headless=False)
ij.getVersion()


# In[31]:


#maskpath = 'C:/Users/User/Desktop/SynthProject/IDE_images/segmaps/5f42a8d4a9.png' #have to be absolute paths
#imagepath = 'C:/Users/User/Desktop/SynthProject/IDE_images/images/5f42a8d4a9.png'
#projectname = 'rods''

def LabelToRoi(maskpath, imagepath, projectname):
    args = {
        'image': maskpath,
        'project': projectname,
     }                                  #imageJ groovy script to transform masks into individual ROIs and saves them into a zip
    language_extension = 'groovy'
    roiscript = """
#@ String image
#@ String project

import ij.io.Opener
import ij.io.OpenDialog
import ij.IJ
import ij.ImagePlus;
import ij.gui.Roi;
import java.util.HashSet;
import ij.process.ImageProcessor;
import ij.plugin.filter.ThresholdToSelection;
import ij.plugin.frame.RoiManager;
import ij.process.FloatProcessor

p = project
imp = new Opener().openImage(image)
rois = labelImageToRoiArray(imp)
putRoisToRoiManager(rois,true);
outputfolder = "C:/Users/User/Desktop/SynthProject/";

    //------------- HELPERS

    public ArrayList<Roi> labelImageToRoiArray(ImagePlus imp) {
        ArrayList<Roi> roiArray = new ArrayList<>();
        ImageProcessor ip = imp.getProcessor();
        float[][] pixels = ip.getFloatArray();

        HashSet<Float> existingPixelValues = new HashSet<>();

        for (int x=0;x<ip.getWidth();x++) {
            for (int y=0;y<ip.getHeight();y++) {
                existingPixelValues.add((pixels[x][y]));
            }
        }

        // Converts data in case thats a RGB Image
        fp = new FloatProcessor(ip.getWidth(), ip.getHeight())
        fp.setFloatArray(pixels)
        imgFloatCopy = new ImagePlus("FloatLabel",fp)

        existingPixelValues.each { v ->
            fp.setThreshold( v,v,ImageProcessor.NO_LUT_UPDATE);
            Roi roi = ThresholdToSelection.run(imgFloatCopy);
            roi.setName(Integer.toString((int) (double) v));
            roiArray.add(roi);
        }
        return roiArray;
    }


    public static void putRoisToRoiManager(ArrayList<Roi> rois, boolean keepROISName) {
        RoiManager roiManager = RoiManager.getRoiManager();
        if (roiManager==null) {
            roiManager = new RoiManager();
        }
            roiManager.reset();
        for (int i = 0; i < rois.size(); i++) {
            if (!keepROISName) {
                rois.get(i).setName(""+i);
            }
            if (rois.get(i).getName() != "0"){ // included this if statement so that the background with label ID=0 is not included in the ROI manager
                roiManager.addRoi(rois.get(i));
            }
        }

        roiManager.runCommand("Save", "./rois.zip");

        roiManager.close();

    }"""

    resultscript = ij.py.run_script('groovy', roiscript, args)

    old_name = "./rois.zip"          #changes the folder zip file name to a project specific one
    new_name = "./"+ projectname +".zip"
    if os.path.isfile(new_name) == False and os.path.isfile(old_name) == True:
        os.rename(old_name, "./"+ projectname +".zip")





# In[25]:


def RoiToPool(maskpath, imagepath, projectname):

    args = {
    'image': maskpath,
    'image2': imagepath,
    'project':f"{projectname}"
    }

    #ImageJ macro to crop all ROIs individually and saves them
    macro = """

#@ String image
#@ String image2
#@ String project
open(image)
run("ROI Manager...");
roiManager("Open", "./"+ project +".zip");
RoiManager.multiCrop("./test_mask_pool/", " save png");
close("ROI Manager")
close()
open(image2)
run("ROI Manager...");
roiManager("Open", "./"+ project +".zip");
RoiManager.multiCrop("./test_image_pool/", " save png");
close()
close("ROI Manager")"""

    old = "./test_mask_pool/"                    #changes the name of the file to a project specific name
    new = "./test_mask_pool_"+ projectname +"/"

    old2 = "./test_image_pool/"
    new2 = "./test_image_pool_"+ projectname +"/"

    if os.path.isfile(old2) == False:
        os.mkdir(old2)
    if os.path.isfile(old) == False:
        os.mkdir(old)

    macroscript = ij.py.run_script('ijm', macro, args)

    if os.path.isfile(new) == False:
        os.rename(old, "./test_mask_pool_"+ projectname +"/") #still have to fix the existing file problem
    if os.path.isfile(new2) == False:
        os.rename(old2, "./test_image_pool_"+ projectname +"/")

    masks = os.listdir(new)              # replaces all other pixels that do not have the same value of the middle one to 0 for all images in the list and overwrite them
    for mask in masks:
        image = cv2.imread(new + mask, -1)
        #print(image.shape)
        h, w = image.shape
        middle_pixel = image.item(round(h/2), round(w/2))
        #print(middle_pixel)
        for y in range(h):
            for x in range(w):
                pixel = image.item(y,x)
                if pixel == middle_pixel:
                    continue
                else: image.itemset((y,x),0)
        cv2.imwrite(new + mask, image)

# In[4]:



import numpy as np

def GetBG(maskpath, imagepath, projectname, only_realbg):                       #generate a system sepcific background image with the mean of bg pixels
    try:
        os.mkdir('./Background/')
    except:
        print('Folder already exists!')
    if only_realbg == False:
        mask_bg = cv2.imread(maskpath, -1)
        black = []
        h,w = mask_bg.shape

        for y in range(h):
            for x in range(w):
                pixel = mask_bg.item(y,x)
                if pixel == 0:
                    black.append((y,x))

        image_bg = cv2.imread(imagepath)
        bg_values = []

        for (y,x) in black:
            px_value = image_bg.item(y,x,1)
            bg_values.append(px_value)

        bg_mean = mean(bg_values)
        #print(round(bg_mean))
        new_bg = np.zeros((1350,1350,3),dtype=np.uint8)
        h_n, w_n, _ = new_bg.shape

        for y in range(h_n):
                for x in range(w_n):
                    new_bg.itemset((y,x,0), round(bg_mean))
                    new_bg.itemset((y,x,1), round(bg_mean))
                    new_bg.itemset((y,x,2), round(bg_mean))
        cv2.imwrite('./Background/background_'+projectname+'.png', new_bg)

    start = Image.open(maskpath).convert("L").save(maskpath)

    input_mask= cv2.imread(maskpath)
    gray_mask = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray_mask,0,255,cv2.THRESH_BINARY)
    image = cv2.imread(imagepath, -1)
    print("Generating BG...")
    image = cv2.inpaint(image, mask, 70, cv2.INPAINT_TELEA)
    image = cv2.resize(image,(1024, 1024),  interpolation = cv2.INTER_AREA)
    bordersize = 163
    image = cv2.copyMakeBorder(image, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    print("Done!")
    cv2.imwrite('./Background/realbackground_'+projectname+'.png', image)
#GetBG(maskpath, imagepath)



# In[32]:


def LabelToPool(maskpath, imagepath, projectname):
    LabelToRoi(maskpath, imagepath, projectname)
    RoiToPool(maskpath, imagepath, projectname)
    return 0


# In[34]:


#LabelToPool(maskpath, imagepath)


# In[ ]:
