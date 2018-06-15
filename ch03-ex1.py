"""
Create a function that takes the image coordinates of a square (or rectangular)
object (for example a book, a poster, or a 2D bar code) and estimates the transform
that takes the rectangle to a full on frontal view in a normalized coordinate system.
Use ginput() or the strongest Harris corners to find the points.
"""

from PIL import Image
from numpy import *
from pylab import *
import sys
import os
from scipy import ndimage

def normalize(points):
  # Normalize homogenous points
  for row in points:
    row /= points[-1]
  return points

def make_homog(points):
  # Make homogenous
  return vstack((points,ones((1, points.shape[1]))))


def H_from_points(fp, tp):
  # Find H such that H * fp = tp.
  # 
  # H has eight degrees of freedom, 
  # so this needs at least 4 points in fp and tp.
  
  if fp.shape != tp.shape:
    raise RuntimeError('number of points do not match')
    #
    # COMPLETE THIS FUNCTION!!
    #
    # --from points--
  m = mean(fp[:2], axis=1)
  maxstd = max(std(fp[:2], axis=1)) + 1e-9
  C1 = diag([1/maxstd, 1/maxstd, 1])
  C1[0][2] = -m[0]/maxstd
  C1[1][2] = -m[1]/maxstd
  fp = dot(C1,fp)
    
  # --to points--
  m = mean(tp[:2], axis=1)
  maxstd = max(std(tp[:2], axis=1)) + 1e-9
  C2 = diag([1/maxstd, 1/maxstd, 1])
  C2[0][2] = -m[0]/maxstd
  C2[1][2] = -m[1]/maxstd
  tp = dot(C2,tp)
    
 # create matrix for linear method, 2 rows for each correspondence pair
  nbr_correspondences = fp.shape[1]
  A = zeros((2*nbr_correspondences,9))
  for i in range(nbr_correspondences):
      A[2*i] = [-fp[0][i],-fp[1][i],-1,0,0,0,
        tp[0][i]*fp[0][i],tp[0][i]*fp[1][i],tp[0][i]]
      A[2*i+1] = [0,0,0,-fp[0][i],-fp[1][i],-1,
        tp[1][i]*fp[0][i],tp[1][i]*fp[1][i],tp[1][i]]

    
  U,S,V = linalg.svd(A)
  H = V[8].reshape((3,3))
    
  # decondition
  H = dot(linalg.inv(C2),dot(H,C1))
   
  return H / H[2, 2]

"""
if len(sys.argv) != 2:
  print 'usage: %prog image.jpeg'
  sys.exit(1)

imname = sys.argv[1]
im = array(Image.open(imname))
"""

im = array(Image.open(os.path.abspath("data/image.jpg")).convert('L'))
imshow(im)
gray()
corners = array(ginput(4))

h=297
w=210

fp_nohomog = transpose(corners)
fp = make_homog(fp_nohomog)
tp = array([[0,0,w,w],[0,h,h,0],[1,1,1,1]])
H = H_from_points(fp,tp)

mapped = ndimage.affine_transform(im,H[:2,:2],(H[0,2],H[1,2]),output_shape=(1500,1500))

figure()
imshow(mapped)
gray()
show()
