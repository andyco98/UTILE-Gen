#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
from random import randrange, uniform
import random
from scipy import ndimage
import os
from PIL import Image, ImageOps, ImageFilter
from main import Texture_Generator
from LabeltoROI import LabelToPool, GetBG
from statistics import mean
from math import sqrt, ceil
import time

########### TO DO ############
# Develope ImageJ substitute
# Generation of Metadata
##############################



def Crop_Best_Rectangle(projectname):

    p_path = "./test_image_pool_"+projectname+"/"
    m_path = "./test_mask_pool_"+projectname+"/"
    p = os.listdir("./test_image_pool_"+projectname+"/")                   #creates the corresponding directories
    m = os.listdir("./test_mask_pool_"+projectname+"/")

    if os.path.isdir('./extracted_textures_'+projectname+'/') == False:
            os.mkdir('./extracted_textures_'+projectname+'/')

    if os.path.isdir('./rgb_mask_pool_'+projectname+'/') == False:
            os.mkdir('./rgb_mask_pool_'+projectname+'/')
    rgb_path = './rgb_mask_pool_'+projectname+'/'

    if os.path.isdir('./rect_image_pool_'+projectname+'/') == False:
            os.mkdir('./rect_image_pool_'+projectname+'/')
    rect_path = './rect_image_pool_'+projectname+'/'
    r = os.listdir(rect_path)

    for i in p:

        mask_tran = Image.open(m_path + i).convert('RGB').save('./rgb_mask_pool_'+projectname+'/'+i)
        #double_mask = Image.open(m_path + i).save(m_path+'real'+i)
        image = cv2.imread(p_path + i , -1)

        mask = cv2.imread(rgb_path + i , -1)
        h, w, _ = mask.shape

        for x in range(h):
            for y in range(w):
                k = mask.item(x,y,0)
                if k != 0:
                    mask.itemset((x,y,0),255)
                    mask.itemset((x,y,1),255)
                    mask.itemset((x,y,2),255)
        result = cv2.bitwise_and(image, mask)

        cv2.imwrite('./rect_image_pool_'+projectname+'/real'+i, result)

    r = os.listdir(rect_path)

    for t,j in enumerate(r):
        try:
            # Import your picture
            input_picture = cv2.imread(rect_path+j)

            # Color it in gray
            gray = cv2.cvtColor(input_picture, cv2.COLOR_BGR2GRAY)

            # Create our mask by selecting the non-zero values of the picture
            ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)

            # Select the contour
            cont, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            # if your mask is incurved or if you want better results,
            # you may want to use cv2.CHAIN_APPROX_NONE instead of cv2.CHAIN_APPROX_SIMPLE,
            # but the rectangle search will be longer

            cv2.drawContours(gray, cont, -1, (255,0,0), 1)


            # Get all the points of the contour
            contour = cont[0].reshape(len(cont[0]),2)

            # we assume a rectangle with at least two points on the contour gives a 'good enough' result
            # get all possible rectangles based on this hypothesis
            rect = []
            #print(len(contour))
            for i in range(len(contour)):
                x1, y1 = contour[i]
                for j in range(len(contour)):
                    x2, y2 = contour[j]
                    area = abs(y2-y1)*abs(x2-x1)
                    rect.append(((x1,y1), (x2,y2), area))

            # the first rect of all_rect has the biggest area, so it's the best solution if he fits in the picture
            all_rect = sorted(rect, key = lambda x : x[2], reverse = True)

            # we take the largest rectangle we've got, based on the value of the rectangle area
            # only if the border of the rectangle is not in the black part

            # if the list is not empty
            if all_rect:

                best_rect_found = False
                index_rect = 0
                nb_rect = len(all_rect)

                # we check if the rectangle is  a good solution
                while not best_rect_found and index_rect < nb_rect:

                    rect = all_rect[index_rect]
                    (x1, y1) = rect[0]
                    (x2, y2) = rect[1]

                    valid_rect = True

                    # we search a black area in the perimeter of the rectangle (vertical borders)
                    x = min(x1, x2)
                    while x <max(x1,x2)+1 and valid_rect:
                        if mask[y1,x] == 0 or mask[y2,x] == 0:
                            # if we find a black pixel, that means a part of the rectangle is black
                            # so we don't keep this rectangle
                            valid_rect = False
                        x+=1

                    y = min(y1, y2)
                    while y <max(y1,y2)+1 and valid_rect:
                        if mask[y,x1] == 0 or mask[y,x2] == 0:
                            valid_rect = False
                        y+=1

                    if valid_rect:
                        best_rect_found = True

                    index_rect+=1

                if best_rect_found:

                    cv2.rectangle(gray, (x1,y1), (x2,y2), (255,0,0), 1)


                    # Finally, we crop the picture and store it
                    result = input_picture[min(y1, y2):max(y1, y2), min(x1,x2):max(x1,x2)]

                    cv2.imwrite("./extracted_textures_"+projectname+"/pa_ex_tx_"+str(j)+".png", result)

                else:
                    print("No rectangle fitting into the area")

            else:
                print("No rectangle found")


        except:
            continue
        print("Loading particle",t + 1, "of",len(r),"(Max 20)")
        if t == 19:
            break

def Particle_To_Texture(particle_image_pool, projectname, only_real):    #extract the textures from the particles

    img_pool = os.listdir(particle_image_pool)
    w_pt = []
    h_pt = []

    for j in img_pool:
        image = Image.open(particle_image_pool+j)
        ws, hs = image.size
        w_pt.append(ws)
        h_pt.append(hs)

    h_mean = mean(h_pt)

    w_mean = mean(w_pt)

    artifact_pt = []

    for c in img_pool:
        image = Image.open(particle_image_pool+c)
        w, h = image.size


        if h < h_mean*0.6:
            artifact_pt.append('./test_image_pool_'+projectname+'/'+c)
            continue

        elif w < w_mean*0.6:
            artifact_pt.append('./test_image_pool_'+projectname+'/'+c)

    image.close()

    for k in artifact_pt:
        os.remove(k)
        filename = os.path.basename(k)
        os.remove('./test_mask_pool_'+projectname+'/'+filename)


    Crop_Best_Rectangle(projectname)


    txt_pool = os.listdir("./extracted_textures_"+projectname+"/")
    print("Generating Texture...")
    if only_real == False:
        if os.path.isdir('./roi_txt_pool_'+projectname+'/') == False:
             os.mkdir('./roi_txt_pool_'+projectname+'/')

        for i,t in enumerate(txt_pool):

            image_name = './extracted_textures_'+projectname+'/'+t

            output_name = './roi_txt_pool_'+projectname+'/tx_'+t

            try:
                Texture_Generator(image_name, output_name, 1)        #generates the textures

            except:
                continue

            print("Texture",i+1, 'of', len(txt_pool), '(max 20) generated successfully!')

            if i == 19:
                break
    print("Done!")




def Crop_Real_To_Patch(projectname): #Transforms and adds the real particles to the particle pool
    masked_pool = './rect_image_pool_'+projectname+'/'
    m = os.listdir(masked_pool)
    q = os.listdir("./test_mask_pool_"+projectname+"/")
    for k in q:
        double_mask = Image.open("./test_mask_pool_"+projectname+"/" + k).save("./test_mask_pool_"+projectname+"/"+"real"+k)
    for s in m:
        img = Image.open(masked_pool+s).convert('RGBA')     #transforms the objects into RGBA and makes transparent
        pixdata = img.load()                                #the surrondings of the object
        width, height = img.size
        for y in range(height):
            for x in range(width):
                if pixdata[x, y] == (0, 0, 0, 255):
                    pixdata[x, y] = (0, 0, 0, 0)
        img.save('./txt_mask_pool_'+projectname+'/'+s)

def Mask_Texture(particle_path, texture_path, projectname, only_real):
    pt_pool = os.listdir(particle_path)
    if os.path.isdir('./txt_mask_pool_'+projectname+'/') == False:
            os.mkdir('./txt_mask_pool_'+projectname+'/')
    if only_real == False:
        for i in pt_pool:
            try:
                img = Image.open(particle_path + i) #transforms the pt masks into RGBA and color the object white for texturizing
                img = img.convert("RGBA")
                img.save('./txt_mask_pool_'+projectname+'/'+i)

                src1 = cv2.imread('./txt_mask_pool_'+projectname+'/'+i, -1)
                h, w , _ = src1.shape
                for y in range(h):
                    for x in range(w):
                        pixel = src1.item(y,x,0)
                        if pixel != 0:
                            src1.itemset((y,x,0), 254)
                            src1.itemset((y,x,1), 254)
                            src1.itemset((y,x,2), 254)

                cv2.imwrite('./txt_mask_pool_'+projectname+'/'+i, src1)


                img = Image.open('./txt_mask_pool_'+projectname+'/'+i)     #transforms the objects into RGBA and makes transparent
                pixdata = img.load()                       #the surrondings of the object
                width, height = img.size
                for y in range(height):
                    for x in range(width):
                        if pixdata[x, y] == (0, 0, 0, 255):
                            pixdata[x, y] = (0, 0, 0, 0)
                img.save('./txt_mask_pool_'+projectname+'/'+i)

                src1 = cv2.imread('./txt_mask_pool_'+projectname+'/'+i, -1)
                txt_pool = random.choice(os.listdir(texture_path))   #Adds the texture to the objects
                src2 = cv2.imread(texture_path+txt_pool)
                src2 = cv2.cvtColor(src2, cv2.COLOR_RGB2RGBA)
                src2 = cv2.resize(src2, src1.shape[1::-1])
                dst = cv2.bitwise_and(src1, src2)
                cv2.imwrite('./txt_mask_pool_'+projectname+'/'+i, dst)
            except:
                continue
    Crop_Real_To_Patch(projectname)        #adds real particles to the pool

def add_salt_and_pepper(image, amount):

    output = np.copy(np.array(image))

    # add salt
    nb_salt = np.ceil(amount * output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_salt)) for i in output.shape]
    output[tuple(coords)] = 1

    # add pepper
    nb_pepper = np.ceil(amount* output.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(nb_pepper)) for i in output.shape]
    output[tuple(coords)] = 0

    image = Image.fromarray(output)
    image = ImageOps.grayscale(image)
    image = image.convert('RGB')
    return image

def NoOverlapCoordinates(dataset_size, projectname):
    m = os.listdir("./txt_mask_pool_"+projectname+"/")
    size = []
    for t in m:
        mask = Image.open("./txt_mask_pool_"+projectname+"/"+t)
        h,w = mask.size
        avg = h+w/2
        size.append(avg)
    dist = ceil(mean(size))

    subsampling = 1

    height, width = 1024, 1024

    row, col = 300, 300
    sq = ceil(sqrt(height/(height/(dataset_size*1.2)))) #number of squares that the whole image is divided
    d= ceil(height/sq)
    check = sq*dist
    coor = []

    if dataset_size > 4:
        for i in range(sq):
            row = subsampling * (i*d + d)
            for j in range(sq):
                col = subsampling * (j*d + d)
                coor.append((row,col))
    else:
        coor = [(ceil(height*0.25),ceil(width*0.25)),(ceil(height*0.25),ceil(width*0.75)),(ceil(height*0.75),ceil(width*0.25)),(ceil(height*0.75),ceil(width*0.75)),(ceil(height*0.5),ceil(width*0.5))]
    return coor


def Random_Paste_Objects(bg_path, pt_path, dataset_size, l, u, projectname, no, sp_amount, gb_amount, min_size, max_size):
    if os.path.isdir('./ground_truth_'+projectname+'/images') == False:
            os.makedirs('./ground_truth_'+projectname+'/images')
    if os.path.isdir('./ground_truth_'+projectname+'/masks') == False:
            os.makedirs('./ground_truth_'+projectname+'/masks')

    for i in range(dataset_size):
        bg = random.choice(os.listdir(bg_path)) #random choice of background and mask bg generation
        background = Image.open(bg_path + bg)
        mask_bg = Image.new('RGB', (1350,1350))
        h_bg, w_bg = background.size
        obj_amount = random.randint(l, u)

        coor = NoOverlapCoordinates(obj_amount, projectname) #Generates regular coordinates for non overlapping systems

        for j in range(obj_amount):
            pt = random.choice(os.listdir(pt_path))
            foreground = Image.open(pt_path + pt)
            rot = random.randint(0, 360)


            w, h = foreground.size                       #object resize
            min_scale = min_size
            max_scale = max_size
            scale = uniform(min_scale,max_scale)
            h_new = int(h*scale)
            w_new = h_new*w/h
            foreground = foreground.resize((int(w_new), int(h_new)), Image.LANCZOS)

            foreground = foreground.rotate(rot, expand = True) #object rot
            #mask
            mask_object = Image.open('./test_mask_pool_'+projectname+'/'+pt).convert('RGBA')

            w_n, h_n = mask_object.size

            pixeldata = mask_object.load()

            for y in range(h_n):
                for x in range(w_n):
                    if pixeldata[x,y] == (0, 0, 0, 255):
                        pixeldata[x, y] = (0, 0, 0, 0)
                    elif pixeldata[x, y] == (0, 0, 0, 0):
                        continue
                    else:
                        pixeldata[x,y] = (j+1, j+1, j+1, 255)


            w_m, h_m = mask_object.size
            h_m_new = int(h_m*scale)
            w_m_new = h_m_new*w_m/h_m
            mask_object = mask_object.resize((int(w_m_new), int(h_m_new)), Image.NEAREST)
            mask_object = mask_object.rotate(rot, expand = True) #object rot
            #Gaussian Blur


            if no:
                regular_position = random.choice(coor)
                coor.remove(regular_position)
                background.paste(foreground, (regular_position), foreground) #paste
                mask_bg.paste(mask_object, (regular_position), mask_object)

            else:
                y_position = random.randint(0, h_bg)
                x_position = random.randint(0, w_bg)
                background.paste(foreground, (y_position, x_position), foreground) #paste
                mask_bg.paste(mask_object, (y_position, x_position), mask_object)

        background = background.crop((163,163,1187,1187))
        mask_bg = mask_bg.crop((163,163,1187,1187))

        #Gaussian blur to full image
        background = add_salt_and_pepper(background, random.uniform(0, sp_amount))
        background = background.filter(ImageFilter.GaussianBlur(radius = random.uniform(0, gb_amount)))

        background.convert("L").save('./ground_truth_'+projectname+'/images/'+projectname+str(i)+".tif")
        mask_bg.convert("L").convert("I;16").save('./ground_truth_'+projectname+'/masks/'+projectname+str(i)+".tif")





def Remove_Zero_Values(input_image_path):

    img = Image.open(input_image_path,'r').convert("L")

    w, h = img.size
    pxdata = img.load()
    for y in range(h):
        for x in range(w):
            if pxdata[x,y] == 0:
                img.putpixel((x,y),2)
            if pxdata[x,y] == 1:
                img.putpixel((x,y),2)

    img.convert('RGB').save(input_image_path)





def Eliminate_Edge_Pt(input_mask_path):
    edge_pt = []
    mask = Image.open(input_mask_path)
    print("Generating mask...")
    w,h = mask.size

    pxdata = mask.load()

    for i in range(w):
        value = pxdata[i,0]
        if value != 0:
            edge_pt.append(value)
    for i in range(w):
        value = pxdata[i,h-1]
        if value != 0:
            edge_pt.append(value)

    for j in range(h):
        value = pxdata[0,j]
        if value != 0:
            edge_pt.append(value)

    for j in range(h):
        value = pxdata[w-1,j]
        if value != 0:
            edge_pt.append(value)

    for q in range(w):
        for r in range(h):
            if pxdata[q,r] in edge_pt:
                mask.putpixel((q,r), 0)
    mask.save("./mask.png")
    input_mask_path = os.path.abspath("mask.png")
    print("Done!")
    return input_mask_path





def Dataset_Gen(dataset_size, l, u, input_mask_path, input_image_path, projectname, no, only_real, only_realbg, sp_amount, gb_amount, min_size, max_size):
    #Generator of image/mask pair
    #max number of particles 250

    #Arguments:
    #1: Dataset size
    #2: Lower limit of particles per image
    #3: Upper limit of particles per image
    #4: Absolute path of the input mask
    #5: Absolute Path of the input image
    #6: Individual project name
    #7: Boolean for non-overlapping systems
    #8: Boolean for using only the real patches (and not the synthetic ones with the generated textures)(faster)
    #9: Boolean for using only real backgrounds
    #10: Maximal percentage of salt-pepper noise added to the image (random between 0 and the given value)
    #11: Maximal percentage of Gaussian Blur added to the image(random between 0 and the given value)
    #12: Minimal scaling factor of the individual objects
    #13: Maximal scaling factor of the individual objects


    path = './'+projectname+'/'

    try:
        os.mkdir(projectname)
    except FileExistsError:
        print('Project already exist! Choose another name!')
        return

    try:
        os.chdir(path)
        #print("Current working directory: {0}".format(os.getcwd()))
    except FileNotFoundError:
        print("Directory: {0} does not exist".format(path))
    except NotADirectoryError:
        print("{0} is not a directory".format(path))
    except PermissionError:
        print("You do not have permissions to change to {0}".format(path))

    Remove_Zero_Values(input_image_path)

    GetBG(input_mask_path, input_image_path, projectname, only_realbg)

    if l > 2:
        input_mask_path = Eliminate_Edge_Pt(input_mask_path)


    LabelToPool(input_mask_path, input_image_path, projectname)

    img_pool = './test_image_pool_'+projectname+'/'

    Particle_To_Texture(img_pool, projectname, only_real)

    pt_pool = './test_mask_pool_'+projectname+'/'
    roi_tx_pool = './roi_txt_pool_'+projectname+'/'


    Mask_Texture(pt_pool, roi_tx_pool, projectname, only_real)

    txt_pt_pool =  './txt_mask_pool_'+projectname+'/'
    bg_path = './Background/'


    Random_Paste_Objects(bg_path, txt_pt_pool, dataset_size, l, u, projectname, no, sp_amount, gb_amount, min_size, max_size)

    orig_img = Image.open(input_image_path)
    orig_img.save("./original_image.png")

    os.chdir('..')





def ConvertFolderRightFormat(folder_path):
    imgs = []
    with os.scandir(folder_path+"/images") as files:
        for file in files:
            imgs.append(file.name)
    try:
        for i in imgs:
            img = Image.open(folder_path+f"/images/{i}").convert("L").save(folder_path+f"/images/{i}.tif")
            msk = Image.open(folder_path+f"/masks/{i}").convert("L").convert("I;16").save(folder_path+f"/masks/{i}.tif")
    except: print("not an image")





def Multiple_Folder_Expansion(dataset_size, l, u, multiple_folder_path, no, only_real, only_realbg, sp_amount, gb_amount, min_size, max_size):
    folder_list = []
    with os.scandir(multiple_folder_path) as folders:
        for folder in folders:
            folder_list.append(folder.name)
    for i in folder_list:
        Dataset_Gen_Folder(dataset_size, l, u, multiple_folder_path+f"/{i}", no, only_real, only_realbg, sp_amount, gb_amount, min_size, max_size)







def Dataset_Gen_Folder(dataset_size, l, u, folder_path, no, only_real, only_realbg, sp_amount, gb_amount, min_size, max_size):
    imgs = []
    with os.scandir(folder_path+"/images") as files:
        for file in files:
            imgs.append(file.name)
    try:
        os.makedirs(folder_path+f"/ground_truth{dataset_size}/images")
        os.mkdir(folder_path+f"/ground_truth{dataset_size}/masks")
    except:
        print("files already exists")
    os.chdir(folder_path)
    for i in imgs:
        input_image_path = folder_path+f"/images/{i}"
        input_mask_path = folder_path+f"/masks/{i}"

        ### Check number of particles in the image ###
        msk = Image.open(input_mask_path)
        msk_here = msk.copy()
        msk.close()
        pt_num = []
        w,h = msk_here.size
        pxdata = msk_here.load()

        for t in range(w):
            for j in range(h):
                value = pxdata[t,j]
                if value != 0 and value not in pt_num:
                    pt_num.append(value)

        number = len(pt_num)
        print(number)

        projectname = i+str(dataset_size)
        Dataset_Gen(dataset_size, number, number, input_mask_path, input_image_path, projectname, no, only_real, only_realbg, sp_amount, gb_amount, min_size, max_size)
        syn_imgs = os.listdir(folder_path+f"/{projectname}/ground_truth_{projectname}/images")
        for name in syn_imgs:
            syn_img = Image.open(folder_path+f"/{projectname}/ground_truth_{projectname}/images/{name}")
            syn_img.save(folder_path+f"/ground_truth{dataset_size}/images/{name}")
        syn_msks = os.listdir(folder_path+f"/{projectname}/ground_truth_{projectname}/masks")
        for name in syn_msks:
            syn_msk = Image.open(folder_path+f"/{projectname}/ground_truth_{projectname}/masks/{name}")
            syn_msk.save(folder_path+f"/ground_truth{dataset_size}/masks/{name}")


def Cluster_Org(cluster): #Takes a cluster find list the images, find the masks and organize them into images/masks folders
    folders = os.listdir(cluster)
    print("folders: ",folders)
    for i in folders:
        if i == '.tmp.driveupload':
            continue
        else:
            path = cluster + "/" + i
            names = []
            os.chdir(path)
            with os.scandir() as files:
                for file in files:
                    names.append(file.name)
            try:
                os.mkdir("images")
                os.mkdir("masks")
            except:
                print("alreadyexist")
            for t in names:
                print(t)
            for t in names:
                if t == "desktop.ini":
                    continue
                else:
                    try:
                        img = Image.open("C:/Users/User/Desktop/SynthProject/IDE_images/images/"+t)
                        img.save(f"./images/{t}")
                        msk = Image.open("C:/Users/User/Desktop/SynthProject/IDE_images/segmaps/"+t)
                        msk.save(f"./masks/{t}")
                    except:
                        print("exception")
        os.chdir("..")
