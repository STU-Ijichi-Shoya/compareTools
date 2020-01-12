import os
import argparse 
## codec err 対策
import codecs
import tensorflow as tf

from PIL import Image
import numpy as np
import hashlib

import warnings
warnings.filterwarnings('ignore')

### TF Recode 仕様:
'''
TF EXAMPLEに各データについての情報があり，それを連結して出力したものがTF Recodeである．
'''




class data:
    height=None
    width=None

    def __init__(self,imagePath,DataPath=None,
                class_label_number=None,
                x_up=None,box_width=None,
                y_up=None,box_height=None,
                ImageHeight=None,ImageWidth=None
                ):
        '''imagePath は画像のパス. DataPathはbool，class_numはクラス番号が入ったテキストのパス．
            注意！！原則としてDataPathの仕様は
            <物体のクラス番号> <検出領域の左上x座標> <検出領域の左上y座標> <検出領域の幅> <検出領域の高さ>とする．
            x_upは左上座標のx座標．box_widthは検出領域の幅．'''
        self.imagePath=imagePath
        
        # self.class_labelName=class_labelName

        ## 拡張子
        _ , self.exh=os.path.splitext(self.imagePath)
        # print(self.exh)
        p=self.imagePath.replace(self.exh,".txt")
        # print("read path",p)
        try:
            if DataPath is None:
                if not (x_up and y_up and box_height and box_width):
                    print("not set ImageData!! please Input DataPath or set data!!")
                    return "er"
                self.x_up=int(x_up)
                self.box_width=int(box_width)

                self.y_up=int(y_up)
                self.box_height=int(box_height)
                DataPath=p
            else:          

                with codecs.open(p,'r', 'utf-8', 'ignore') as f:
                    t=f.read()
                    
                t=t.split()
                self.class_num,self.x_up,self.y_up,self.box_width,self.box_height=list(map(int,t))
                DataPath=p
            
            self.class_labelName=os.path.dirname(self.imagePath).split("/")[-1]
            print(self.class_labelName)
        except:
            import traceback
            traceback.print_exc()

            print("fail",self.imagePath)



        if self.height is self.width is None:
            if ImageHeight is None or ImageWidth is None:
                raise("not set ImageSize!!")
            self.height=ImageHeight
            self.width=ImageWidth

def createTF_Example(data:data):
    width = int(data.width)
    height = int(data.height)

    filename = str(data.class_num)
    

    # with tf.gfile.GFile(data.imagePath, 'rb') as fid:
    #     encoded_image_data = fid.read()
    encoded_image_data = np.array(Image.open(data.imagePath)).tostring()
    image_format = data.exh.encode("utf-8")
    
    key = hashlib.sha256(encoded_image_data).hexdigest()

    xmins = []
    xmaxs = []
    ymins = [] 
    ymaxs = []
    classes_text = []
    classes = []
    # for obj in example_dict['object']:
    #     xmins.append(float(obj['bndbox']['xmin']) / width)
    #     ymins.append(float(obj['bndbox']['ymin']) / height)
    #     xmaxs.append(float(obj['bndbox']['xmax']) / width)
    #     ymaxs.append(float(obj['bndbox']['ymax']) / height)
    #     class_name = obj['name']
    #     classes_text.append(class_name.encode('utf8'))
    #     classes.append(label_map_dict[class_name])

    classes_text=[data.class_labelName.encode("utf-8")]
    classes=[data.class_num]

    xmins=[data.x_up/width]
    ymins=[data.y_up/height]
    ymaxs=[data.box_height/width]
    xmaxs=[data.box_width/width]



    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_image_data])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text':  tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

def getArg():
    pars=argparse.ArgumentParser(description="<物体のクラス番号> <検出領域の左上x座標> <検出領域の左上y座標> <検出領域の幅> <検出領域の高さ>の入った画像情報と画像から，TFRecode(train用,test用)を作成．")
    pars.add_argument("-i","--imagesDirctory",help="画像のディレクトリ(labelName/imagepath)，また，imagepathと同じディレクトリに同じファイル名の画像情報が入ったディレクトリを想定している.つまりanydirctory/labelName/imagepathの場合anydirctry(絶対パス)を指定する．",required=True)
    pars.add_argument("-e","--extention",help="画像ファイルの拡張子,デフォルトはbmp.",default="bmp")
    pars.add_argument("-sw","--width",help="画像の幅を指定する．デフォルトは200",default=200,required=False)
    pars.add_argument("-sh","--height",help="画像の高さを指定する．デフォルトは200",default=200,required=False)
    pars.add_argument("-o","--outputDirctory",help="TF-Recodeの出力ディレクトリ デフォルトはカレントディレクトリ",default=os.getcwd(),required=False)
    pars.add_argument("-v","--validationRate",help="テストデータと訓練データを分ける割合．デフォルトのテストデータ割合は0.3",default=0.3,required=False)
    return pars.parse_args()



def main():
    import glob,multiprocessing.pool as pool
    import random

    #load arg
    arg=getArg()

    p=os.path.join(arg.imagesDirctory,"**","*"+arg.extention)
    
    pathlist=glob.glob(p,recursive=True)
    
    ## seed setting 
    random.seed(114514)
    random.shuffle(pathlist)

    trainlen=int(len(pathlist)*float(arg.validationRate))
    
    p=pool.Pool()
    train_classlist=[data(p,True,ImageHeight=arg.height,ImageWidth=arg.width) for p in pathlist[:trainlen]]
    test_classlist=[data(p,True,ImageHeight=arg.height,ImageWidth=arg.width) for p in pathlist[trainlen::]]

    
    rtrain=p.map(createTF_Example,train_classlist)
    rtest=p.map(createTF_Example,test_classlist)
    
    trainfilepath=os.path.join(arg.outputDirctory,"trainFiles.tfrecode")
    testfilepath=os.path.join(arg.outputDirctory,"testFiles.tfrecode")

    w=tf.python_io.TFRecordWriter(trainfilepath)
    for i in rtrain:
        w.write(i.SerializeToString())
    w.close()

    w=tf.python_io.TFRecordWriter(testfilepath)
    for i in rtest:
        w.write(i.SerializeToString())
    w.close()
    
    print("finished. len(train)=%d len(test)=%d" % (trainlen,len(pathlist)-trainlen) )
    



if __name__ == "__main__":
    main()

    
