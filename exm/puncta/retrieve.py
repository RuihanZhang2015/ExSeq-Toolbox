import h5py
import os 
import numpy as np
import pickle


def retrieve_complete(self,fov):
    with open(self.args.puncta_path+'/fov{}/complete.pkl'.format(fov),'rb') as f:
        return pickle.load(f)
    
def retrieve_coordinate(self):
    with open(self.args.layout_file,encoding='utf-16') as f:
        contents = f.read()

        contents = contents.split('\n')
        contents = [line for line in contents if line and line[0] == '#' and 'SD' not in line]
        contents = [line.split('\t')[1:3] for line in contents]

        coordinate = [[float(x) for x in line] for line in contents ]
        coordinate = np.asarray(coordinate)

        coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
        coordinate[:,1] -= min(coordinate[:,1])
        coordinate = np.round(np.asarray(coordinate/0.1625/(0.90*2048))).astype(int)
        return coordinate

def retrieve_coordinate2(self):
    import xml.etree.ElementTree 
    def get_offsets(filename= "/mp/nas3/fixstars/yves/zebrafish_data/20221025/code2/stitched_raw_small.xml"):
        tree = xml.etree.ElementTree.parse(filename)
        root = tree.getroot()
        vtrans = list()
        for registration_tag in root.findall('./ViewRegistrations/ViewRegistration'):
            tot_mat = np.eye(4, 4)
            for view_transform in registration_tag.findall('ViewTransform'):
                affine_transform = view_transform.find('affine')
                mat = np.array([float(a) for a in affine_transform.text.split(" ")] + [0, 0, 0, 1]).reshape((4, 4))
                tot_mat = np.matmul(tot_mat, mat)
            vtrans.append(tot_mat)
        def transform_to_translate(m):
            m[0, :] = m[0, :] / m[0][0]
            m[1, :] = m[1, :] / m[1][1]
            m[2, :] = m[2, :] / m[2][2]
            return m[:-1, -1]

        trans = [transform_to_translate(vt).astype(np.int64) for vt in vtrans]
        return np.stack(trans)

    coordinate = get_offsets()
    coordinate = np.asarray(coordinate)

    coordinate[:,0] = max(coordinate[:,0]) - coordinate[:,0]
    coordinate[:,1] -= min(coordinate[:,1])
    coordinate[:,2] -= min(coordinate[:,2])
    # coordinate[:,:2] = np.asarray(coordinate[:,:2]/0.1625)
    # coordinate[:,2] = np.asarray(coordinate[:,2]/0.4)
    return coordinate
