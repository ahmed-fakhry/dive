import caffe
import numpy as np
import sys

if len(sys.argv) != 3:
    print "Usage: python convert_protomean.py proto.mean out.npy"
    sys.exit()

np.set_printoptions(threshold='nan')
blob = caffe.proto.caffe_pb2.BlobProto()
data = open( sys.argv[1] , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
print out
np.save( sys.argv[2] , out )
f = open('mean.txt', 'w')
f.write(str(out))
f.close()