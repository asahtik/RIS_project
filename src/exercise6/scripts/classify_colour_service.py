import cv2
import random
import numpy as np
import rospy
from exercise6 import RecogniseColour, RecogniseColourRequest, RecogniseColourResponse

NEAREST_K = 1

class KNN:
  def __init__(self):
    # Cylinders
    r = open("knn_data/cyl_red.txt")
    rdata = [l.split() for l in r.readlines()]
    r.close()

    g = open("knn_data/cyl_green.txt")
    gdata = [l.split() for l in g.readlines()]
    g.close()

    y = open("knn_data/cyl_yellow.txt")
    ydata = [l.split() for l in y.readlines()]
    y.close()

    b = open("knn_data/cyl_blue.txt")
    bdata = [l.split() for l in b.readlines()]
    b.close()

    red = random.sample(rdata, 100)
    green = random.sample(gdata, 100)
    yellow = random.sample(ydata, 100)
    blue = random.sample(bdata, 100)

    for i in red:
      i.append(1)
    for i in green:
      i.append(2)
    for i in yellow:
      i.append(3)
    for i in blue:
      i.append(4)

    cdata = np.array(red + green + yellow + blue)
    clabels = cdata[:,-1]
    cdata = cdata[:, :-1]
    self.knnc = cv2.ml.KNearest_create()
    self.knnc.train(cdata, cv2.ml.ROW_SAMPLE, clabels)

    # Rings
    r = open("knn_data/ring_red.txt")
    rdata = [l.split() for l in r.readlines()]
    r.close()

    g = open("knn_data/ring_green.txt")
    gdata = [l.split() for l in g.readlines()]
    g.close()

    bc = open("knn_data/ring_black.txt")
    bcdata = [l.split() for l in bc.readlines()]
    bc.close()

    b = open("knn_data/ring_blue.txt")
    bdata = [l.split() for l in b.readlines()]
    b.close()

    red = random.sample(rdata, 100)
    green = random.sample(gdata, 100)
    black = random.sample(bcdata, 100)
    blue = random.sample(bdata, 100)

    for i in red:
      i.append(1)
    for i in green:
      i.append(2)
    for i in black:
      i.append(5)
    for i in blue:
      i.append(4)

    rdata = np.array(red + green + black + blue)
    rlabels = rdata[:,-1]
    rdata = rdata[:, :-1]
    self.knnr = cv2.ml.KNearest_create()
    self.knnr.train(rdata, cv2.ml.ROW_SAMPLE, rlabels)

model = KNN()

def handle_recognise(req):
  hist = np.array(req.hist)
  t = req.type
  if t == req.CYLINDER:
    ret, results, neighbours, dist = model.knnc.findNearest(hist, NEAREST_K)
    return RecogniseColourResponse(colour=results[0][0])
  elif t == req.RING:
    ret, results, neighbours, dist = model.knnr.findNearest(hist, NEAREST_K)
    return RecogniseColourResponse(colour=results[0][0])
  else:
    return RecogniseColourResponse(colour=0)

if __name__ == "__main__":
  rospy.init_node('colour_recognition_server')
  s = rospy.Service('recognise_colour', RecogniseColour, handle_recognise)

