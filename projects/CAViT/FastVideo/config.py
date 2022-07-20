from fastreid.config import CfgNode as CN

def add_temp_config(cfg):
    _C = cfg

    _C.TEMP = CN()
    _C.TEMP.REDUCE = 'avg'

    _C.TEMP.DATA = CN()
    _C.TEMP.DATA.DENSE          = False
    _C.TEMP.DATA.SAMPLING_STEP  = 32

    _C.TEMP.TRAIN = CN()
    _C.TEMP.TRAIN.SAMPLER = 'TemporalRestrictedCrop'
    _C.TEMP.TRAIN.SEQ_SIZE = 8
    _C.TEMP.TRAIN.STRIDE        = 4

    _C.TEMP.TEST = CN()
    _C.TEMP.TEST.ALL = False
    _C.TEMP.TEST.SAMPLER = 'TemporalRestrictedBeginCrop'
    _C.TEMP.TEST.SEQ_SIZE       = 4
    _C.TEMP.TEST.STRIDE       = 8
    _C.TEMP.TEST.TRACK_SPLIT    = 64

    _C.TEMP.CLUSTER = CN()
    _C.TEMP.CLUSTER.NUM = 5
    _C.TEMP.CLUSTER.EPOCH = 5
    _C.TEMP.CLUSTER.FRAMES = 16
    _C.TEMP.CLUSTER.DATA_RATIO = 0.5


def add_reweight_config(cfg):
    _C = cfg
    _C.REWEIGHT = CN()
    _C.REWEIGHT.EPOCH = 50
    _C.REWEIGHT.UPPER = 0.85
    _C.REWEIGHT.LOWER = 0.6


def add_swin_config(cfg):

    _C = cfg
    
    _C.MODEL.BACKBONE.PATCH_SIZE   = 4
    _C.MODEL.BACKBONE.PADDING_SIZE = (0, 1)
    _C.MODEL.BACKBONE.WINDOW_SIZE  = 4
    _C.MODEL.BACKBONE.QKV_BIAS     = True
    _C.MODEL.BACKBONE.QK_SCALE     = None
    _C.MODEL.BACKBONE.MLP_RATIO    = 4.
    _C.MODEL.BACKBONE.APE          = False
    _C.MODEL.BACKBONE.PATCH_NORM   = True



def add_vit_config(cfg):

    _C = cfg

    _C.MODEL.BACKBONE.SHIFT_NUM        = 5
    _C.MODEL.BACKBONE.SHUFFLE_GROUP    = 2
    _C.MODEL.BACKBONE.DEVIDE_LENGTH    = 4
    _C.MODEL.BACKBONE.RE_ARRANGE       = True
    _C.MODEL.BACKBONE.INFERENCE_DEPTH  = 12
    _C.MODEL.BACKBONE.LAYER_NUM        = 12
    _C.MODEL.BACKBONE.NORM_OUT         = False
    _C.MODEL.BACKBONE.PART_POOL        = 'max'
    _C.MODEL.BACKBONE.TOKEN            = 'all'
    _C.MODEL.BACKBONE.PADDING_SIZE     = (0, 0)
    _C.MODEL.BACKBONE.NUM_CAMERA       = 0
    _C.MODEL.BACKBONE.SEQ_MAX           = 1000
    _C.MODEL.BACKBONE.T_DEPTH           = 2
    _C.MODEL.BACKBONE.ATT_TYPE          = 'divided_space_time'  #["divided_space_time", "space_only", ""]
    _C.MODEL.BACKBONE.NUM_FRAMES        = 8
    _C.MODEL.BACKBONE.CONVE_SHARE       = False
    _C.MODEL.BACKBONE.DILATION          = 1
    _C.MODEL.BACKBONE.PATCH_SIZE        = (16, 16)


def add_cascade_config(cfg):
    _C = cfg

    _C.CASCADE = CN()

    _C.CASCADE.PATCH1  = (16, 16)
    _C.CASCADE.STRIDE1 = (16, 16)

    _C.CASCADE.PATCH2  = (16, 32)
    _C.CASCADE.STRIDE2 = (16, 32)

    _C.CASCADE.PATCH3  = (32, 16)
    _C.CASCADE.STRIDE3 = (32, 16)

    _C.CASCADE.TPE = 'flow'
    _C.CASCADE.MAX_LEN = 1000


def add_slowfast_config(cfg):
    _C = cfg

    
    _C.SLOW = CN()
    _C.SLOW.FRAME_NUM   = 4
    _C.SLOW.IMG_SIZE    = (384, 192)
    _C.SLOW.PATCH       = (16, 16)
    _C.SLOW.STRIDE      = (16, 16)


    _C.FAST = CN()
    _C.FAST.FRAME_NUM   = 8
    _C.FAST.IMG_SIZE    = (256, 128)
    _C.FAST.PATCH       = (16, 16)
    _C.FAST.STRIDE      = (16, 16)



