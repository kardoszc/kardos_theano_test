import os
import Image
import pickle

faces_root = '/mnt/hgfs/kardos_test/theano/face_predict/mit/faces/'
nonfaces_root = '/mnt/hgfs/kardos_test/theano/face_predict/mit/nonfaces/'
def load_pickel(media_root, label):
    data = []
    for file_name in os.listdir(media_root):
    	print file_name
        try :
            im = Image.open(os.path.join(media_root+file_name))
            pixel = []
            for i in range(20):
                for j in range(20):
                    p = im.getpixel((i,j))
                    p = p/256.0 if type(p) == int else sum(p)/len(p)/256.0
                    pixel.append(p)
            data.append([file_name, pixel, label])
        except:
            continue
    return data

data = {}
for d in load_pickel(faces_root, 1) + load_pickel(nonfaces_root, 0):
	data[d[0]] = d[1:]

data_list = [[],[],[]]
for key in data.keys():
	data_list[0].append(key)
	data_list[1].append(data[key][0])
	data_list[2].append(data[key][1])
output = open('faces.pkl', 'wb')
pickle.dump(data_list, output)
output.close()
