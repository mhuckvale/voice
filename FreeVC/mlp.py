# basic MLP execution for MLP-V2 format networks
#
import numpy as np
#
class MLP:
	def __init__(self):
		self.numinput = 0
		self.numhidden = [0, 0]
		self.numoutput = 0
		self.weights = {}
		self.flags = 0
	#
	# load the MLP in MLP-V2 format
	def load(self,mlpfile):
		with open(mlpfile) as ip:
			header=ip.readline().rstrip()
			if (header!="MLP-V2"):
				print("error reading %s" % (mlpfile))
				return;
			line=ip.readline().rstrip().replace("="," ").split()
			if (line[0]!="NUMINPUTS"):
				print("error reading %s" % (mlpfile))
				return;
			self.numinput = int(line[1]);
			line=ip.readline().rstrip().replace("="," ").split()
			if (line[0]!="NUMHIDDEN"):
				print("error reading %s" % (mlpfile))
				return;
			self.numhidden = [ int(line[1]), int(line[2]) ]
			line=ip.readline().rstrip().replace("="," ").split()
			if (line[0]!="NUMOUTPUTS"):
				print("error reading %s" % (mlpfile))
				return;
			self.numoutput = int(line[1])
			line=ip.readline().rstrip().replace("="," ").split()
			if (line[0]!="flags"):
				print("error reading %s" % (mlpfile))
				return;
			self.flags = int(line[1])
			# read in weights from input to first layer
			self.weights["layer1"]=np.zeros((self.numhidden[0],1+self.numinput),dtype=float)
			for i in range(self.numhidden[0]):
				line=ip.readline().rstrip().replace("=","").split()
				self.weights["layer1"][i,:]=np.array(line[1:],dtype="float32");
			# read in weights from layer 1 to layer 2
			self.weights["layer2"]=np.zeros((self.numhidden[1],1+self.numhidden[0]),dtype=float)
			for i in range(self.numhidden[1]):
				line=ip.readline().rstrip().replace("=","").split()
				self.weights["layer2"][i,:]=np.array(line[1:],dtype="float32");
			# read in weights from layer 2 to output
			self.weights["layer3"]=np.zeros((self.numoutput,1+self.numhidden[1]),dtype=float)
			for i in range(self.numoutput):
				line=ip.readline().rstrip().replace("=","").split()
				self.weights["layer3"][i,:]=np.array(line[1:],dtype="float32");

	def sigmoid(self,x):
		return 1 / (1 + np.exp(-x))

	# run a vector through the network
	def forward(self,x):
		layer1=self.weights["layer1"][:,0]+np.dot(self.weights["layer1"][:,1:],x[:]);
		#print(layer1.shape);
		layer1=np.tanh(layer1);
		layer2=self.weights["layer2"][:,0]+np.dot(self.weights["layer2"][:,1:],layer1[:]);
		#print(layer2.shape);
		layer2=np.tanh(layer2);
		layer3=self.weights["layer3"][:,0]+np.dot(self.weights["layer3"][:,1:],layer2[:]);
		#print(layer3.shape);
		return(np.float32(layer3));

	# run a set of vectors
	def run(self,xtest):
		ypred=np.zeros((xtest.shape[0],self.numoutput),dtype=float);
		for i in range(xtest.shape[0]):
			ypred[i,:]=self.forward(xtest[i,:]);
		return ypred
