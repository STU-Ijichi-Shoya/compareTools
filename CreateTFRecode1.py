import warnings
warnings.filterwarnings('ignore')
import os
import argparse 
## codec err 対策
import codecs

from PIL import Image
import numpy as np
import hashlib
from io import BytesIO
from PIL import Image
import collections

### TF Recode 仕様:
'''
TF EXAMPLEに各データについての情報があり，それを連結して出力したものがTF Recodeである．
'''




class data:
    height=None
    width=None
    class_num_dict={}
    
    def __init__(self,imagePath,DataPath=None,
                class_label_number=None,
                x_up=None,box_width=None,
                y_up=None,box_height=None,
                ImageHeight=None,ImageWidth=None,
		is_converted=False
                ):

        '''imagePath は画像のパス. DataPathはbool，class_numはクラス番号が入ったテキストのパス．
            注意！！原則としてDataPathの仕様は
            <物体のクラス番号> <検出領域の左上x座標> <検出領域の左上y座標> <検出領域の幅> <検出領域の高さ>とする．
            x_upは左上座標のx座標．box_widthは検出領域の幅．'''
        self.imagePath=imagePath
        ImageHeight,ImageWidth=int(ImageHeight),int(ImageWidth)

        ## 拡張子
        _ , self.exh=os.path.splitext(self.imagePath)
        # print(self.exh)
        p=self.imagePath.replace(self.exh,".txt")
        # print("read path",p)
        try:   
            with codecs.open(p,'r', 'utf-8', 'ignore') as f:
                t=f.read()
                t=t.split()
                self.class_num,self.x_up,self.y_up,self.box_width,self.box_height=list(map(float,t))
                ## must !=0 0 class is background only !!
                self.class_num=int(self.class_num)+1
            self.DataPath=p

            self.class_labelName=os.path.dirname(self.imagePath).split("/")[-1]
            self.class_num_dict[self.class_labelName]=self.class_num
        except:
            import traceback
            traceback.print_exc()

            print("fail",self.imagePath)



        if self.height is self.width is None:
            if ImageHeight is None or ImageWidth is None:
                raise("not set ImageSize!!")
            self.height=ImageHeight
            self.width=ImageWidth
        
        if not is_converted:
	        self.x_up/=self.width
	        self.box_width/=self.width
	        self.y_up/=self.height
	        self.box_height/=self.height
    def get_class_num_name(self):
        return self.class_num_dict
	
def createPBtext(itemnum:int,itemName)->str:
    ss="""item{
        id: %d
        name: \'%s\'
    }\n"""%(itemnum,itemName)
    return ss

def createTF_Example(data:data):
    import tensorflow as tf

    width = int(data.width)
    height = int(data.height)

    filename = str(data.class_num)
    
    # with tf.gfile.GFile(data.imagePath, 'rb') as fid:
    #     encoded_image_data = fid.read()
    with BytesIO() as bs:
        img=Image.open(data.imagePath)
        img.save(data.imagePath+".jpeg",format="jpeg")
    with tf.gfile.GFile(data.imagePath+".jpeg", 'rb') as fid:
        encoded_jpg = fid.read()

    image_format = b"jpg"
    

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

    xmins=[data.x_up]
    ymins=[data.y_up]
    ymaxs=[data.box_height]
    xmaxs=[data.box_width]



    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
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
    pars.add_argument("-c","--converted",help="",default=True,required=False)

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
    print("Files found.")
    p=pool.Pool()
    train_classlist=[data(p,True,ImageHeight=arg.height,ImageWidth=arg.width) for p in pathlist[:trainlen]]
    test_classlist=[data(p,True,ImageHeight=arg.height,ImageWidth=arg.width) for p in pathlist[trainlen::]]

    
    rtrain=p.map(createTF_Example,train_classlist)
    rtest=p.map(createTF_Example,test_classlist)
    print("prosessing...")
    trainfilepath=os.path.join(arg.outputDirctory,"testFiles.recode")
    testfilepath=os.path.join(arg.outputDirctory,"trainFiles.recode")
    pbtxtfilepath=os.path.join(arg.outputDirctory,"pbtext.txt")
    
    import tensorflow as tf
    print("write files...")
    w=tf.python_io.TFRecordWriter(trainfilepath)
    for i in rtrain:
        w.write(i.SerializeToString())
    w.close()

    w=tf.python_io.TFRecordWriter(testfilepath)
    for i in rtest:
        w.write(i.SerializeToString())
    w.close()
    
    with open(pbtxtfilepath,"w") as w:
        ll=train_classlist[0].get_class_num_name()
        ss=""
        for index in ll:
            ss+=createPBtext(ll[index],index)
        print(ss,file=w)

    print("finished. len(test)=%d len(train)=%d" % (trainlen,len(pathlist)-trainlen) )
    



if __name__ == "__main__":
    main()

    
