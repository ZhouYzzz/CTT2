import os.path as osp

ROOT_DIR = osp.join(osp.dirname(__file__), '..')

def read_sequence(base, sqs_name):
    '''return: (groundtruth<np.array>, img_iter<list(String)>) '''
    sqs_path = osp.join(ROOT_DIR,base,sqs_name)
    if osp.exists(sqs_path):
        import numpy as np
        groundtruth = np.loadtxt(
            osp.join(sqs_path,'groundtruth_rect.txt'),
            delimiter=','
            )
        img_path = osp.join(sqs_path,'img')
        img_count = groundtruth.shape[0]
        img_iter = list()
        for i in xrange(1, img_count+1):
            frame = img_path + '/%04d.jpg'%i
            img_iter.append(frame)
            if not osp.exists(frame):
                raise IOError('Missing Frame %d'%i)

        return groundtruth, img_iter
    else:
        raise IOError('No Target Sequence')

if __name__ == '__main__':
    groundtruth, img_iter = read_sequence('benchmark','MotorRolling')