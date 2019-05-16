"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

from yolo3.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])#残差相加
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)#最后深度都是512
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    #这里是一个尺度，所以num_anchors = 3
    #最小尺度的输出，全卷积，应该是13x13？但是看网络结构图此处是8x8，最后输出是out_filters深度

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)#深度变256，向上采样变16x16
    x = Concatenate()([x,darknet.layers[152].output])#大小不变，深度叠加上darknet.layers[152]层的深度
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))
    #中尺度，看网络结构是16*16*num_anchors*(num_classes+5)

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)
    x = Concatenate()([x,darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs, [y1,y2,y3])

def tiny_yolo_body(inputs, num_anchors, num_classes):
    '''Create Tiny YOLO_v3 model CNN body in keras.'''
    x1 = compose(
            DarknetConv2D_BN_Leaky(16, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(32, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(64, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(128, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(256, (3,3)))(inputs)
    x2 = compose(
            MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),
            DarknetConv2D_BN_Leaky(512, (3,3)),
            MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'),
            DarknetConv2D_BN_Leaky(1024, (3,3)),
            DarknetConv2D_BN_Leaky(256, (1,1)))(x1)
    y1 = compose(
            DarknetConv2D_BN_Leaky(512, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))(x2)

    x2 = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x2)
    y2 = compose(
            Concatenate(),
            DarknetConv2D_BN_Leaky(256, (3,3)),
            DarknetConv2D(num_anchors*(num_classes+5), (1,1)))([x2,x1])

    return Model(inputs, [y1,y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    #注意！！！！这个函数是对一个batch_size所有的图片同时处理的，通过向量化
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    '''
    参数变换
    [[[[[116.  90.]
        [156. 198.]
        [373. 326.]]]]]
    '''
    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])# shape：[grid_shape[0],grid_shape[1],1,1]
    #K.tile()在某一维度上重复多少次
    #K.arange(0, stop = 13)构造[0~12]列表
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])# shape：[grid_shape[1],grid_shape[0],1,1]
    grid = K.concatenate([grid_x, grid_y])# shape:[grid,grid,1,2]
                                          #例：如果是最后一层13x13，则构成[13,13,1,2]的栅格网络，保存每个网格的坐标从(0,0)~(13,13)
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    # 将box_xy，box_xy 从OUTPUT的预测数据转为标准尺度的坐标(应该是416,416)。
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))#转化为相对于grid的xy坐标
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))#转化为相对于grid的宽高
    box_confidence = K.sigmoid(feats[..., 4:5])#置信度回归用
    box_class_probs = K.sigmoid(feats[..., 5:])#类别概率 也是用于回归

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#转换成适配不同图片本身的box尺寸
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    '''Get corrected boxes'''
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])#这个应该是m个图片并行操作
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    '''Process Conv layer output'''
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats,
        anchors, num_classes, input_shape)
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs#最后得分是(置信度*分类器)的概率
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    """Evaluate YOLO model on given input and return filtered boxes."""
    num_layers = len(yolo_outputs)#yolo_outputs应该是[y1,y2,y3]这样的格式
                                  #其中y：[m,gird,grid,3,5+1]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]] # default setting
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32#(416,416)
    boxes = []
    box_scores = []  #shape:[某一尺度所有图片每一个grid，类别数],保存的是每个类别的概率*置信度，也就是说置信度为0就为0
    for l in range(num_layers):#三个尺度分别转化为实际画框的参数
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
            anchors[anchor_mask[l]], num_classes, input_shape, image_shape)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)#有点像降维，m张图同时操作

    mask = box_scores >= score_threshold#分数满足的掩膜
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        # TODO: use keras backend instead of tf.
        class_boxes = tf.boolean_mask(boxes, mask[:, c])#清理达不到置信度*类别概率阈值的的类别，第一次清理
                                                        #即清理所有gird预测的box中，没有物体的，概率特别小的box
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)#非极大值抑制，最多留20个！
        class_boxes = K.gather(class_boxes, nms_index)#通过下标找到该box
        class_box_scores = K.gather(class_box_scores, nms_index)#通过下表找到该box的分数
        classes = K.ones_like(class_box_scores, 'int32') * c#把类型变成一个整数而非'00...1...000'的形式
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)

    return boxes_, scores_, classes_


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(m, T, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), wh
    num_classes: integer

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''
    #true_boxes.shape  = (图片张数，每张图片box个数，5)（5是左上右下点坐标加上类别下标）
    assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'#判断类别是否超出了20
    num_layers = len(anchors)//3 # default setting此处的anchors为9
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]

    true_boxes = np.array(true_boxes, dtype='float32')#shape(图片张数，每张图片box个数，5)
    input_shape = np.array(input_shape, dtype='int32')#[416 416] shape(2,)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2#将每个box的左上点和右下点坐标相加除2，即取中点！
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]#将每个box的(x2-x1,y2-y1)，即宽和高，
                                                           #尺寸为(图片张数，每张图box个数,2)2:宽和高
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]#将中心点及宽高 对输入图片416 做归一化
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    m = true_boxes.shape[0]#图片张数
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]#获取特征图的尺寸 13， 26,52 
    #[(13,13), (26,26), (52,52)]
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
  #    '''
  #   y_true是一个长度为3的列表，列表包含三个numpyarray float32类型的全零矩阵，具体形状如下
  #   (6, 13, 13, 3, 25) (6, 26, 26, 3, 25) (6, 52, 52, 3, 25) 即三个尺度特征图大小     
  #   '''
    #[(m,13,13,3,5+num_classes),(m,26,26,3,5+num_classes),(m,52,52,3,5+num_classes)] #3是每个grid预测三个bbox

    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)#9个anchor扩展维度,扩展第一个维度原来为(9,2) --->(1,9,2)这样操作可以充分利用numpy的广播机制
    anchor_maxes = anchors / 2.     #将anchors 中心点放（0,0） 因为anchors没有中心点只有宽高，计算与boxs计算iou时两者中心点均为（0.0）
    anchor_mins = -anchor_maxes#这两行貌似是将anchor的位置移到x轴上下
    valid_mask = boxes_wh[..., 0]>0#掩膜尺寸:[图片张数，每张图box个数] 即每个box都有一个0或1的掩膜来掩w宽
                                    #(2,3)的矩阵里面存着True或False
                                    #判断是否有异常标注boxes 

    for b in range(m):
        # Discard zero rows.
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        # Expand dim to apply broadcasting.
        wh = np.expand_dims(wh, -2)#变成(box个数,1,2)

        box_maxes = wh / 2.
        box_mins = -box_maxes#跟上面的anchor一样移动一下位置，且尺寸相同
 
        #很显然是计算真实值和anchor的IOU
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each true box
        #通过iou最大的确定该BOX应该放在Label的哪个anchor的位置
        best_anchor = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:#看一下best_anchor在哪个尺度
                    i = np.floor(true_boxes[b,t,0]*grid_shapes[l][1]).astype('int32')#这两个表示是哪一个grid预测
                    j = np.floor(true_boxes[b,t,1]*grid_shapes[l][0]).astype('int32')#即用true的x,y乘上grid_shapes
                    k = anchor_mask[l].index(n)#应该放在该尺度该grid三个anchor的哪个位置上
                    c = true_boxes[b,t, 4].astype('int32')#是哪一个类别的
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b,t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1#将分类器的结果变成0和1的形式，分到的那类是1

    return y_true
    #[(None,13,13,3,5+num_classes),(None,26,26,3,5+num_classes),(None,52,52,3,5+num_classes)]
    #None是不知道会有多少张图在这个尺度

def box_iou(b1, b2):
    '''Return iou tensor

    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)

    '''

    # Expand dim to apply broadcasting.
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=False):
    #args前三个元素为yolov3输出的预测值，后三个维度为保存的label 值
    '''Return yolo_loss tensor

    Parameters
    ----------
    yolo_outputs: list of tensor, the output of yolo_body or tiny_yolo_body
    y_true: list of array, the output of preprocess_true_boxes
    anchors: array, shape=(N, 2), wh
    num_classes: integer
    ignore_thresh: float, the iou threshold whether to ignore object confidence loss

    Returns
    -------
    loss: tensor, shape=(1,)

    '''
    num_layers = len(anchors)//3 # default setting

    #args即[*model_body.output, *y_true]
    #model_body.output = [y1,y2,y3]即三个尺度的预测结果,每个y都是m*grid*grid*num_anchors*(num_classes+5)
    #m = batch_size
    yolo_outputs = args[:num_layers]
    y_true = args[num_layers:]
    anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if num_layers==3 else [[3,4,5], [1,2,3]]
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))#得到(416*416)
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    #得到三个grid的大小
    loss = 0
    m = K.shape(yolo_outputs[0])[0] # batch size, tensor
    mf = K.cast(m, K.dtype(yolo_outputs[0]))#mf为batchsize大小
    #逐层计算损失 
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]#置信率
        true_class_probs = y_true[l][..., 5:]#分类

        #将网络最后一层输出转化为BBOX的参数
        #anchors[anchor_mask[l]]:anchors对应的某一个尺度的anchor
        #例：最小尺度预测大物体：
        '''
        anchors[anchor_mask[0]]
        [[116  90]
        [156 198]
        [373 326]]
        '''
        #yolo_head将预测的偏移量转化为真实值，这里的真实值是用来计算iou,并不是来计算loss的，loss使用偏差来计算的
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l],
             anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=True)#anchor_mask[0]=[6,7,8] 
        pred_box = K.concatenate([pred_xy, pred_wh])#相对于gird的box参数(x,y,w,h)
                                                    #anchors[anchor_mask[l]]=array([[  116.,   90.],
                                                                                 # [  156., 198.],
                                                                                 # [  373., 326.]])

        # Darknet raw box to calculate loss.
        #这是对x,y,w,b转换公式的反变换
        raw_true_xy = y_true[l][..., :2]*grid_shapes[l][::-1] - grid#保存时其实保存的是5个数(:2)就是x,y
                                                                    #根据公式将boxes中心点x,y的真实值转换为偏移量
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])#计算宽高的偏移量
        #这部操作是避免出现log(0) = 负无穷，故当object_mask置信率接近0是返回全0结果
        #K.switch(条件函数，返回值1，返回值2)其中1,2要等shape
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]#（2-box_ares）避免大框的误差对loss 比小框误差对loss影响大

        # Find ignore mask, iterate over each of batch.
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)#定义一个size可变的张量来存储不含有目标的预测框的信息
        object_mask_bool = K.cast(object_mask, 'bool')#将真实标定的数据置信率转换为T or F的掩膜，映射成bool类型  1=true 0=false
        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])#b是第几张图，将置信率为0的其他参数清0
            iou = box_iou(pred_box[b], true_box)#单张图片单个尺度算iou，一张图片预测出的所有boxes与所有的ground truth boxes计算iou 计算过程与生成label类似利用了广播特性这里不详细描述
            best_iou = K.max(iou, axis=-1)#先取每个grid最大的iou
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))#删掉小于阈值的BBOX
            return b+1, ignore_mask
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])#毅种循环
        ignore_mask = ignore_mask.stack()#将一个列表中维度数目为R的张量堆积起来形成维度为R+1的新张量
        ignore_mask = K.expand_dims(ignore_mask, -1)
        #当一张图片的最大IOU低于ignore_thresh，则认为图片内是没有目标。
        #这里保存的应该是iou满足条件的BBOX

        # K.binary_crossentropy is helpful to avoid exp overflow.
        #x,y交叉熵损失，首先要置信度不为0
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[...,0:2], from_logits=True)
        #宽高损失，方差损失？
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-raw_pred[...,2:4])
        #置信度损失，交叉熵，这里没有物体的部分也要计算损失
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True)+ \
            (1-object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        #分类的损失
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)

        #计算一个batch的总损失
        xy_loss = K.sum(xy_loss) / mf# mf：batch_size
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)], message='loss: ')
    return loss
