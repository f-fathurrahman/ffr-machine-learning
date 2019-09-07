import numpy as np

class Perceptron:
    """ A basic Perceptron"""

    def __init__(self, inputs, targets):
        
        # Set up network size
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else: 
            self.nIn = 1
    
        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
    
        # Initialise network
        self.weights = np.random.rand(self.nIn+1, self.nOut)*0.1 - 0.05

    def __str__(self):
        strs  = "-------------------\n"
        strs += "Perceptron instance\n"
        strs += "-------------------\n"
        strs += "nIn   = {}\n".format(self.nIn)
        strs += "nOut  = {}\n".format(self.nOut)
        strs += "nData = {}\n".format(self.nData)
        strs += "Initial weights: shape = {}\n".format(np.shape(self.weights))
        for w in self.weights:
            strs += "w = {}\n".format(w)
        return strs

    def train(self, inputs, targets, eta, nIterations):
        """ Train the network """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
        # Training
        change = range(self.nData)

        for n in range(nIterations):
            
            self.activations = self.forward(inputs);
            self.weights -= eta*np.dot(np.transpose(inputs),self.activations-targets)
        
            # Randomise order of inputs
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
        #return self.weights

    def forward(self, inputs):
        """ Run the network forward """
        # Compute activations
        activations =  np.dot(inputs,self.weights)

        # Threshold the activations
        return np.where(activations > 0, 1, 0)

    def confusion_matrix(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.nData,1))), axis=1)
        print(inputs)
        
        outputs = np.dot(inputs,self.weights)
        print(outputs)

        nClasses = np.shape(targets)[1]

        if nClasses==1:
            nClasses = 2
            outputs = np.where(outputs>0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses,nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i,j] = np.sum( np.where(outputs==i,1,0)*np.where(targets==j, 1, 0))

        print(cm)
        print(np.trace(cm)/np.sum(cm))
        

def test_logic():

    """ Run AND and XOR logic functions"""
    a = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
    b = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

    p = Perceptron(a[:,0:2],a[:,2:])
    print(p)
    p.train(a[:,0:2],a[:,2:],0.25,10)
    p.confusion_matrix(a[:,0:2],a[:,2:])

    #q = Perceptron(b[:,0:2], b[:,2:])
    #q.train(b[:,0:2], b[:,2:], 0.25, 10)
    #q.confusion_matrix(b[:,0:2], b[:,2:])

test_logic()