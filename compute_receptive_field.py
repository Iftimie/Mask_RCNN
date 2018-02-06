# [filter size, stride, padding]
#Assume the two dimensions are the same
#Each kernel requires the following parameters:
# - k_i: kernel size
# - s_i: stride
# - p_i: padding (if padding is uneven, right padding will higher than left padding; "SAME" option in tensorflow)
#
#Each layer i requires the following parameters to be fully represented:
# - n_i: number of feature (data layer has n_1 = imagesize )
# - j_i: distance (projected to image pixel distance) between center of two adjacent features
# - r_i: receptive field of a feature in layer i
# - start_i: position of the first feature's receptive field in layer i (idx start from 0, negative means the center fall into padding)

import math

#AUTO-ENCODER
convnet,layer_names = [],[]
# stage 1
convnet.append([7, 2, 3])# more channels for this one
layer_names.append('conv1')
# convnet.append([3, 2, 1])
# layer_names.append('C1-pool')

#c = no. conv blocks, i = no. identity blocks, s = stride for conv blocks
stages = {
             'S2':{'c':1,'i':0,'s':2}, #more channels
             'S3':{'c':1,'i':0,'s':2}, #more channels
             'S4':{'c':1,'i':0,'s':2},#or 22 #more channels
             #'S5':{'c':1,'i':0,'s':2}
}
#No batch-normalization

for s in sorted(stages.iterkeys()):
    for x in range(stages[s]['c']):
        stride = stages[s]['s']
        convnet.append([3, stride, 1])
        layer_names.append(s + "-" + "B" + chr(97+ x) +"-2a")
        convnet.append([3, 1, 1])
        layer_names.append(s + "-" + "B" + chr(97+ x) +"-2b")
        # convnet.append([1, 1, 0])
        # layer_names.append(s + "-" + "B" + chr(97+ x) +"-2c")

    for x in range(stages[s]['i']):
        # convnet.append([1, 1, 0])
        # layer_names.append(s + "-" + "B" + chr(97+ x) +"-2a")
        convnet.append([3, 1, 1])
        layer_names.append(s + "-" + "B" + chr(97+ x) +"-2b")
        # convnet.append([1, 1, 0])
        # layer_names.append(s + "-" + "B" + chr(97+ x) +"-2c")


# convnet =   [[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1],[3,2,0],[6,1,0], [1, 1, 0]]
# layer_names = ['conv1','pool1','conv2','pool2','conv3','conv4','conv5','pool5','fc6-conv', 'fc7-conv']
imsize = 128

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]

    n_out = math.floor((n_in - k + 2*p)/s) + 1
    actualP = (n_out-1)*s - n_in + k
    pR = math.ceil(actualP/2)
    pL = math.floor(actualP/2)

    j_out = j_in * s
    r_out = r_in + (k - 1)*j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
    print(layer_name + ":")
    print("\t n features: %s \n \t jump: %s \n \t receptive size: %s \t start: %s " % (layer[0], layer[1], layer[2], layer[3]))

layerInfos = []
if __name__ == '__main__':
    #first layer is the data layer (image) with n_0 = image size; j_0 = 1; r_0 = 1; and start_0 = 0.5
    print ("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])
    print ("------------------------")
    layer_name = raw_input ("Layer name where the feature in: ")
    layer_idx = layer_names.index(layer_name)
    idx_x = int(raw_input ("index of the feature in x dimension (from 0)"))
    idx_y = int(raw_input ("index of the feature in y dimension (from 0)"))

    n = layerInfos[layer_idx][0]
    j = layerInfos[layer_idx][1]
    r = layerInfos[layer_idx][2]
    start = layerInfos[layer_idx][3]
    assert(idx_x < n)
    assert(idx_y < n)

    print ("receptive field: (%s, %s)" % (r, r))
    print ("center: (%s, %s)" % (start+idx_x*j, start+idx_y*j))