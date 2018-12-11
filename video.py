import cv2
import os
import argparse

#Get directory name
parser=argparse.ArgumentParser(description='Remote Driving')
parser.add_argument(
    'image_folder',
    type=str,
    nargs='?',
    default='',
    help='Path to image folder. This is where the images from the run will be saved.'
)
args=parser.parse_args()
dir=args.image_folder
if  dir == '':
    print('Provide a directory of images')
    exit()

#get image file names
filenames=os.listdir(dir)
#remove output file if exists already
if os.path.exists(dir+'.mp4'):
    os.remove(dir+'.mp4')

absOutVidPath=os.path.join(os.getcwd(),dir+'.mp4')
out=None

for file in filenames:#should be proper order since name is timestamp should be sorted from lowest(earliest) to highest(latest) time
    absImgPath=os.path.join(os.getcwd(),dir,file)
    img=cv2.imread(absImgPath,1)
    if out is None:#create VideoWriter object to create video file
        out=cv2.VideoWriter(absOutVidPath,cv2.VideoWriter_fourcc(*'MP4V'),60,(img.shape[1],img.shape[0]))
    out.write(img)
out.release()
