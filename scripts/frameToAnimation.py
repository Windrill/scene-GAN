#import cv2
# Will be cleaned after notes are taken.
import os
import numpy as np
import glob
import moviepy.editor as mpy
'''
def cevaluatePath(in_path, name):
  folders = [x.name for x in os.scandir(in_path) if not (x.name.startswith('.'))]
  for f in folders:
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = np.resize(image, (rows, cols, 1))
    #image = (image)/127.5-1
    #pred = self.generator.predict(np.reshape(image, (1, rows, cols, 1, channels)))
    print(in_path, f)
    os.makedirs(os.path.join(in_path, f), exist_ok = True)
    path = os.path.join(in_path, f, str(name))+".mp4"
    print("pp", path)
    # if image size does not match cv2 will not write at all
    ouou = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 32, (80, 45))
    
    # cv2 fails silently if image is not opened:
    # opened = ouou.isOpened()
    for frame in range(1,33):
      predFrame = cv2.imread(os.path.join(in_path, f,str(frame))+".png")
      ##print((np.asarray(predFrame)).shape)
      #predFrame = np.asarray(((pred[0,:,:,frame-1,:]+1)*127.5).astype(np.uint8))
      ##predFrame = cv2.cvtColor(predFrame, cv2.COLOR_GRAY2RGB)
      ouou.write(np.asarray(predFrame))
    ouou.release()
    cv2.destroyAllWindows()
'''
in_path = "C:/Users/Penblader/Desktop/eval nineteen examples/"

def cwrap(func, path):
  folds = [x for x in glob.glob(path+str("/*")) if os.path.isdir(x) and not x.startswith('.')]
  #folds = [x for x in os.scandir(path)]
  #not (x.name.startswith('.')) and if os.path.isdir(x.name)
  #aa = folds[0]
  #os.path.isdir(aa), TypeError: _isdir: illegal type for path parameter
  #only the third is true...
  #print(aa, os.path.isdir(aa.name), os.path.isdir(os.path.join(path, aa.name)), os.path.join(path, aa.name))
  print("c",path)
  print("cc",folds)
  func(path)
  for fold in folds:
    cwrap(func, fold)

def mp2gif(path):
  mps = [x.name for x in os.scandir(path) if x.name.endswith('.mp4')]
  for mp in mps:
    clip = mpy.VideoFileClip(os.path.join(path,mp))
    clip.show()
    respath = os.path.join(path, str("gout.gif"))
    
    print(respath)
    clip.write_gif(respath)
    
  
if __name__ == '__main__':
  func = mp2gif
  cwrap(func, in_path)
  #cevaluatePath(in_path, "out")