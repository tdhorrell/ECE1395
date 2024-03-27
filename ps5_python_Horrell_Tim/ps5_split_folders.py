#ps5_split_folders.py

import os
import re
import shutil
import random
from PIL import Image

#starting directory and destination directories
directory = 'input\\all'
trainDest = 'input\\train'
testDest = 'input\\test'
personID = ''
imageID = ''
randInd = random.sample(range(1,11), 2)

#---------------------------------------------------------
#                 Rename Files and Sort
#---------------------------------------------------------

#iterate through each folder
for filename in os.listdir(directory):
    folders = os.path.join(directory, filename)
    #keep person id to name later
    personID = re.sub(r'[^\d]+', '', folders)
    
    if os.path.isdir(folders):
        #iterate through each file in each folder
        for imagename in os.listdir(folders):
            image = os.path.join(folders, imagename)
            #image ID to rename
            imageID = re.sub(r'[^\d]+', '', imagename)
            tempName = str(folders + '\\' + personID + '_' + imageID)
            tempNamePng = str(tempName + '.png')
            os.rename(image, tempName)

            #save file as png
            new_file = "{}.png".format(tempName)
            with Image.open(tempName) as im:
                im.save(new_file)

            #sort to directory in accordance with random Index
            if((int(imageID) == randInd[0]) or (int(imageID) == randInd[1])):
                shutil.copy(tempNamePng, testDest)
            else:
                shutil.copy(tempNamePng, trainDest)