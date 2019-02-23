
import pickle
import numpy as np
import tensorflow as tf

import DataLoader
import EmbedHelper


# LRP Methods




# Confidence Methods
def KL(alpha,outputSize):
    beta=tf.constant(np.ones((1,outputSize)),dtype=tf.float32)
    S_alpha = tf.reduce_sum(alpha,axis=1,keep_dims=True)
    S_beta = tf.reduce_sum(beta,axis=1,keep_dims=True)
    lnB = tf.lgamma(S_alpha) - tf.reduce_sum(tf.lgamma(alpha),axis=1,keep_dims=True)
    lnB_uni = tf.reduce_sum(tf.lgamma(beta),axis=1,keep_dims=True) - tf.lgamma(S_beta)
    
    dg0 = tf.digamma(S_alpha)
    dg1 = tf.digamma(alpha)
    
    kl = tf.reduce_sum((alpha - beta)*(dg1-dg0),axis=1,keep_dims=True) + lnB + lnB_uni
    return kl

def mse_loss(p, alpha, global_step, annealing_step,outputSize): 
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
    E = alpha - 1
    m = alpha / S
    
    A = tf.reduce_sum((p-m)**2, axis=1, keep_dims=True) 
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keep_dims=True) 
    
    annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * KL(alp,outputSize)
    return (A + B) + C

# loss_EDL(self.outputsOht, self.alpha, self.global_step, self.annealing_step, outputSize)
def loss_EDL(p, alpha, global_step, annealing_step, outputSize):
    S = tf.reduce_sum(alpha, axis=1, keep_dims=True) 
    E = alpha - 1

    A = tf.reduce_mean(tf.reduce_sum(p * (tf.digamma(S) - tf.digamma(alpha)),1, keepdims=True))

    annealing_coef = tf.minimum(1.00,tf.cast(global_step/annealing_step,tf.float32))

    alp = E*(1-p) + 1 
    B =  annealing_coef * KL(alp,outputSize)

    return (A + B)


class PyramidCNNVShort:
    
    hiddenSize = 250
    
    def __init__(self,inputSize,vectorSize,outputSize):
        hiddenSize = self.hiddenSize
        
        self.paperGraph = tf.Graph()
        with self.paperGraph.as_default():

            self.initializer = tf.contrib.layers.variance_scaling_initializer()

            self.nn_inputs = tf.placeholder(tf.string,[None,inputSize])
            self.nn_vector_inputs = tf.placeholder(tf.float32,[None,inputSize,vectorSize])

            self.token_lengths = tf.placeholder(tf.int32,[None])

            self.nn_outputs = tf.placeholder(tf.int32,[None])
            
            self.annealing_step = tf.placeholder(dtype=tf.int32) 
            
            self.uncertaintyRatio = tf.placeholder(dtype=tf.float32)
            
            self.outputsOht = tf.one_hot(self.nn_outputs,outputSize)

            self.isTraining = tf.placeholder(tf.bool, name="PH_isTraining")     

            self.fullInputs = self.nn_vector_inputs
            fullVectorSize = self.fullInputs.shape[2]

            print("fullvectorsize: ", fullVectorSize)
            

            num_filters = hiddenSize
            self.cnnInput = tf.expand_dims(self.fullInputs, -1, name="absolute_input")

            convouts = []

            # filter_shape = [3, 300, 1, num_filters] #3-4-5, 300, 1, 128
            # W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
            # b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
            # self.conv1 = tf.nn.conv2d(self.cnnInput,W2,strides=[1, 1, 1, 1],padding="VALID", name="conv")

            self.conv1 = tf.layers.conv2d(self.cnnInput,hiddenSize,(3,fullVectorSize),(1,1),padding="VALID",activation=None,use_bias=True,name="PreBlock")       
            self.blockPool = tf.nn.max_pool(self.conv1,ksize=[1,inputSize-3+1,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool-"))

            print(self.conv1.shape)
            
            
            # self.conv2 = tf.layers.conv2d(self.cnnInput,hiddenSize,(4,fullVectorSize),(1,1),padding="valid",activation=None,use_bias=True,name="PreBlock2")
            # self.blockPool2 = tf.nn.max_pool(self.conv2,ksize=[1,125,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool1-"))
            
            # self.conv3 = tf.layers.conv2d(self.cnnInput,hiddenSize,(5,fullVectorSize),(1,1),padding="valid",activation=None,use_bias=True,name="PreBlock3")
            # self.blockPool3 = tf.nn.max_pool(self.conv3,ksize=[1,124,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool2-"))
            
            convouts.append(self.blockPool)
            # convouts.append(self.blockPool2)
            # convouts.append(self.blockPool3)
            
            # print("-")
            # print("conv1 shape :",self.conv1.shape)
            # print("blockPool shape : ",self.blockPool.shape)
            
            # print("-")
            # print("conv2 shape :",self.conv2.shape)
            # print("blockPool2 shape : ",self.blockPool2.shape)
            
            # print("-")
            # print("conv2 shape :",self.conv3.shape)
            
            # print("blockPool3 shape : ",self.blockPool3.shape)

            num_filters_total = 1*hiddenSize
            # num_filters_total = 3*hiddenSize

            # print("convouts : ",convouts)
            self.h_pool = tf.concat(convouts,1)
            # print("hpool : ",h_pool.shape)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

            # print("h_pool_flat : ",h_pool_flat.shape)

            self.h_drop = tf.layers.dropout(self.h_pool_flat, 0.5, training=self.isTraining)



            with tf.name_scope("fully-connected"):
                fc1_neurons = 150
                fc2_neurons = 100
                fc3_neurons = 25

                self.fc1 = tf.layers.dense(self.h_drop,activation=tf.nn.leaky_relu,name="fc1",use_bias=True,kernel_initializer=self.initializer,units=fc1_neurons)
                self.fcD1 = tf.layers.dropout(self.fc1,0.5,training=self.isTraining)


            with tf.name_scope("output"):
                self.scores = tf.layers.dense(self.fcD1,activation=None,name="logits",use_bias=True,kernel_initializer=self.initializer,units=outputSize)
                self.evidence = tf.exp(self.scores/1000)
            
            with tf.name_scope("evaluation"):
                self.global_step = tf.Variable(0,trainable=False)
                init_learn_rate = 0.001
                decay_learn_rate = tf.train.exponential_decay(init_learn_rate,self.global_step,100,0.90,staircase=True)
                
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.optimizer = tf.train.AdamOptimizer()
                
            
            # print(outputsOht.shape)
            # print(tf.argmax(outputsOht, 1).shape)
            self.predictions = tf.argmax(self.scores, 1, name="absolute_output")
            self.truths = tf.argmax(self.outputsOht, 1)
            self.correct_predictions = tf.equal(self.predictions, self.truths)
            # print(self.correct_predictions.shape)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
            self.match = tf.reshape(tf.cast(tf.equal(self.predictions, self.truths), tf.float32),(-1,1))
                
                
                
            with tf.name_scope("uncertainty"):
                self.alpha = self.evidence +1

                self.uncertainty = outputSize / tf.reduce_sum(self.alpha, axis=1, keep_dims=True) #uncertainty

                self.prob = self.alpha/tf.reduce_sum(self.alpha, 1, keepdims=True) 

                total_evidence = tf.reduce_sum(self.evidence ,1, keepdims=True) 
                mean_ev = tf.reduce_mean(total_evidence)
                self.mean_ev_succ = tf.reduce_sum(tf.reduce_sum(self.evidence ,1, keepdims=True)*self.match) / tf.reduce_sum(self.match+1e-20)
                self.mean_ev_fail = tf.reduce_sum(tf.reduce_sum(self.evidence ,1, keepdims=True)*(1-self.match)) / (tf.reduce_sum(tf.abs(1-self.match))+1e-20) 
            
            
                flatUncertainty = tf.reshape(self.uncertainty,shape=[-1,1])
                flatCP = tf.reshape(self.correct_predictions,shape=[-1,1])
                
                # print("Flat Uncertainty : ",flatUncertainty.shape)
                # print("Flat CP :",flatCP.shape)
                zeros = tf.cast(tf.zeros_like(flatUncertainty),dtype=tf.bool)
                ones = tf.cast(tf.ones_like(flatUncertainty),dtype=tf.bool)
                ucAccuraciesBool = tf.where(tf.less_equal(flatUncertainty,self.uncertaintyRatio),ones,zeros)
                
                # print("Mask : ",ucAccuraciesBool.shape)
                self.ucAccuracies = tf.boolean_mask(flatCP,ucAccuraciesBool)
                
                self.ucAccuracy = tf.reduce_mean(tf.cast(self.ucAccuracies,"float"))
                
                self.dataRatio = tf.shape(self.ucAccuracies)[0] / tf.shape(flatCP)[0]
            
            with tf.name_scope("loss"):
                # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.outputsOht)
                # self.loss = tf.reduce_mean(losses)
                
                self.loss = tf.reduce_mean(loss_EDL(self.outputsOht, self.alpha, self.global_step, self.annealing_step,outputSize))
                regLoss = tf.add_n([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()])       
                regularazationCoef = 0.0000005   
                self.loss += regLoss*regularazationCoef

            with tf.control_dependencies(update_ops):
                # self.train_op = optimizer.minimize(self.loss,global_step=global_step,colocate_gradients_with_ops=True)
                self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)


def __init__(self,classDict,embedHandler,dataHandler,inputLength,nnModel):
        self.classDict = classDict
        self.dataHandler = dataHandler
        self.embedHandler = embedHandler
        self.inputLength = inputLength
        self.nnModel = nnModel
        
        self.sess = tf.Session(graph=nnModel.paperGraph)
        with nnModel.paperGraph.as_default():
            with self.sess.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, "/tmp/model.ckpt")


class Predicter:
    EmbeddingsFolderPath = "Embeddings"
    ModelsFolderPath = "NNModels"
    SystemDataFolderPath = "SystemData"
    
    def __init__(self):
        with open(Predicter.SystemDataFolderPath+'/fold0classDict.pkl', 'rb') as f:
            self.classDict =pickle.load(f)
        self.dataHandler = DataLoader.DataHandler

        self.inputLength = 128

        self.loadModels(Predicter.ModelsFolderPath)

        self.nnModel = None
        self.glove = None
        self.htWord2Vec = None
        self.pubmed = None
        self.googleNews = None

        self.initializeEmbeddings()
    
    def loadModels(self,folderPath):
        self.googleNewsNN,self.googleNewsSession = self.getModel(folderPath+"/GoogleNews/model.ckpt",300)
        self.htWord2VecNN,self.htWord2VecSession = self.getModel(folderPath+"/HTW2V/model.ckpt",300)
        self.gloveNN,self.gloveSession = self.getModel(folderPath+"/Glove-Confidence/model.ckpt",300)
        self.pubmedNN,self.pubmedSession = self.getModel(folderPath+"/Pubmed/model.ckpt",200)

        self.nnModel = self.gloveNN
        self.sess = self.gloveSession

    def getModel(self,path,vectorSize):
        with tf.device("/cpu:0"):
            nnModel = PyramidCNNVShort(128, vectorSize, 12)
            sess = tf.Session(graph=nnModel.paperGraph)
            with nnModel.paperGraph.as_default():
                with sess.as_default():
                    saver = tf.train.Saver()
                    saver.restore(sess, path)
        return nnModel,sess


    def initializeEmbeddings(self):
        self.htWord2Vec = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[3], False, 300,
                                                       Predicter.EmbeddingsFolderPath)
        self.googleNews = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[2], False, 300,
                                                       Predicter.EmbeddingsFolderPath)
        self.glove = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[5], False, 300, Predicter.EmbeddingsFolderPath)
        self.pubmed = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[4], False, 200, Predicter.EmbeddingsFolderPath)
        self.embedHandler = self.glove
                
    
        
    def predict(self,ip,lang):
        if(lang=="TR"):
            engData = self.dataHandler.translateInput(ip)
            ip = engData

        cleanData,length = self.dataHandler.inputPreprocessor([ip],self.inputLength)
        vecData = self.embedHandler.vectorizeBatch(cleanData)
        
        feedDict = {self.nnModel.nn_vector_inputs:vecData,self.nnModel.isTraining:False}
        uncertainty,prob = self.sess.run([self.nnModel.uncertainty,self.nnModel.prob],feed_dict=feedDict)
        return uncertainty,prob,length
    
    def predictCOP(self,ip,lang):
        uncertainty,prob,length = self.predict(ip,lang)
        uncertainty = uncertainty[0]
        prob = prob[0]
        
        reverseClassDict = {value:key for key,value in self.classDict.items()}
        if (lang == "TR"):
            reverseClassDict = {key: DataLoader.labelTranslateDict[value] for key, value in reverseClassDict.items()}
        outputClassCount = len(reverseClassDict)
        
        probDict = {reverseClassDict[i]:prob[i] for i in np.arange(outputClassCount)}
        probMatrix = []
        for i in range(len(prob)):
            probMatrix.append([reverseClassDict[i], prob[i]])

        probMatrix = sorted(probMatrix, key=lambda x: (x[1]), reverse=True)

        maxIdx = np.argmax(prob)
        resultDict = {
            "Uncertainty":uncertainty[0],
            "Confidence":1-uncertainty[0],
            "Prediction":reverseClassDict[maxIdx],
            "PredictionProb":prob[maxIdx],
            "Probabilities":probMatrix,
            "Length":length
        }
        
        return resultDict
    
    def predictModel(self,ip,embeddingType,lang="ENG"):
        reverseClassDict = {value:key for key,value in self.classDict.items()}

        self.checkEmbeddings(embeddingType)

        if(embeddingType == "Glove"):
            self.embedHandler = self.glove
            self.nnModel = self.gloveNN
            self.sess = self.gloveSession

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[3]):
            self.embedHandler = self.htWord2Vec
            self.nnModel = self.htWord2VecNN
            self.sess = self.htWord2VecSession

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[2]):
            self.embedHandler = self.googleNews
            self.nnModel = self.googleNewsNN
            self.sess = self.googleNewsSession

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[4]):
            self.embedHandler = self.pubmed
            self.nnModel = self.pubmedNN
            self.sess = self.pubmedSession
        else:
            raise Exception("Embedding Given DOESNT exist")


        predDict = self.predictCOP(ip,lang)
        predProb = predDict["Probabilities"]
        predDict["Top3"] = predProb[0:3]

        return predDict

    def checkEmbeddings(self,embeddingType):
        if(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[5]):
            if (self.glove is None):
                self.glove = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[5], False, 300, Predicter.EmbeddingsFolderPath)

        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[3]):
            if(self.htWord2Vec is None):
                self.htWord2Vec = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[3], False, 300,
                                                               Predicter.EmbeddingsFolderPath)
        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[2]):
            if(self.googleNews is None):
                self.googleNews = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[2], False, 300,
                                                               Predicter.EmbeddingsFolderPath)
        elif(embeddingType == EmbedHelper.EmbeddingHandler.embedDict[4]):
            if(self.pubmed is None):
                self.pubmed = EmbedHelper.EmbeddingHandler(EmbedHelper.EmbeddingHandler.embedDict[4], False, 200,
                                                               Predicter.EmbeddingsFolderPath)






class myModel_CNN_TEXT:
    
    hiddenSize = 128
    
    def __init__(self,inputSize,vectorSize,outputSize):
        hiddenSize = self.hiddenSize
        
        self.paperGraph = tf.Graph()
        with self.paperGraph.as_default():

            self.initializer = tf.contrib.layers.xavier_initializer()
            self.nn_inputs = tf.placeholder(tf.string,[None,inputSize])
            self.nn_vector_inputs = tf.placeholder(tf.float32,[None,inputSize,vectorSize])
            self.token_lengths = tf.placeholder(tf.int32,[None])
            self.nn_outputs = tf.placeholder(tf.int32,[None]) 
            self.outputsOht = tf.one_hot(self.nn_outputs,outputSize)
            self.isTraining = tf.placeholder(tf.bool, name="PH_isTraining")     
            self.fullInputs = self.nn_vector_inputs


            fullVectorSize = self.fullInputs.shape[2]
            num_filters = hiddenSize

            self.cnnInput = tf.expand_dims(self.fullInputs, -1, name="absolute_input")

            convouts = []

            filter_shape = [3, 300, 1, num_filters] #3-4-5, 300, 1, 128
            W2 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W2")
            b2 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b2")
            self.conv1 = tf.nn.conv2d(self.cnnInput,W2,strides=[1, 1, 1, 1],padding="VALID", name="conv")     
            self.blockPool = tf.nn.max_pool(self.conv1,ksize=[1,inputSize-3+1,1,1],strides=[1,1,1,1],padding="VALID",name=("Pool-"))     

            convouts.append(self.blockPool)
            num_filters_total = 1*hiddenSize

            self.h_pool = tf.concat(convouts,1)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            self.h_drop = tf.layers.dropout(self.h_pool_flat, 0.5, training=self.isTraining)


            with tf.name_scope("output"):
                W3 = tf.get_variable(
                    "W3",
                    shape=[num_filters_total, outputSize],
                    initializer=tf.contrib.layers.xavier_initializer())
                b3 = tf.Variable(tf.constant(0.1, shape=[outputSize]), name="b3")
                self.scores = tf.nn.xw_plus_b(self.h_drop, W3, b3, name="scores")
                # self.scores = tf.layers.dense(self.fcD1,activation=None,name="logits",use_bias=True,kernel_initializer=self.initializer,units=outputSize)
                self.evidence = tf.exp(self.scores/1000)
            
            with tf.name_scope("evaluation"):
                self.global_step = tf.Variable(0, trainable=False)        
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.optimizer = tf.train.AdamOptimizer()
             



            self.predictions = tf.argmax(self.scores, 1, name="absolute_output")


            with tf.name_scope("uncertainty"):
                self.alpha = self.evidence +1

                self.uncertainty = outputSize / tf.reduce_sum(self.alpha, axis=1, keep_dims=True) #uncertainty

                self.prob = self.alpha/tf.reduce_sum(self.alpha, 1, keepdims=True) 

                total_evidence = tf.reduce_sum(self.evidence ,1, keepdims=True) 
                mean_ev = tf.reduce_mean(total_evidence)
                self.mean_ev_succ = tf.reduce_sum(tf.reduce_sum(self.evidence ,1, keepdims=True)*self.match) / tf.reduce_sum(self.match+1e-20)
                self.mean_ev_fail = tf.reduce_sum(tf.reduce_sum(self.evidence ,1, keepdims=True)*(1-self.match)) / (tf.reduce_sum(tf.abs(1-self.match))+1e-20) 
            
            
                flatUncertainty = tf.reshape(self.uncertainty,shape=[-1,1])
                flatCP = tf.reshape(self.correct_predictions,shape=[-1,1])
                
                # print("Flat Uncertainty : ",flatUncertainty.shape)
                # print("Flat CP :",flatCP.shape)
                zeros = tf.cast(tf.zeros_like(flatUncertainty),dtype=tf.bool)
                ones = tf.cast(tf.ones_like(flatUncertainty),dtype=tf.bool)
                ucAccuraciesBool = tf.where(tf.less_equal(flatUncertainty,self.uncertaintyRatio),ones,zeros)
                
                # print("Mask : ",ucAccuraciesBool.shape)
                self.ucAccuracies = tf.boolean_mask(flatCP,ucAccuraciesBool)
                
                self.ucAccuracy = tf.reduce_mean(tf.cast(self.ucAccuracies,"float"))
                
                self.dataRatio = tf.shape(self.ucAccuracies)[0] / tf.shape(flatCP)[0]


            with tf.name_scope("loss"):
                # Calculate mean cross-entropy loss
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.outputsOht)
                self.loss = tf.reduce_mean(losses) # + l2_reg_lambda * l2_loss

                self.loss = tf.reduce_mean(loss_EDL(self.outputsOht, self.alpha, self.global_step, self.annealing_step, outputSize))
                regLoss = tf.add_n([tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()])       
                regularazationCoef = 0.0000005   
                self.loss += regLoss*regularazationCoef
            

            self.truths = tf.argmax(self.outputsOht, 1)
            self.correct_predictions = tf.equal(self.predictions, self.truths)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
            self.match = tf.reshape(tf.cast(tf.equal(self.predictions, self.truths), tf.float32),(-1,1))
    
            
            

            with tf.control_dependencies(update_ops):
                # self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
                # self.train_op = self.optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)
                self.train_op = self.optimizer.minimize(self.loss,global_step=self.global_step,colocate_gradients_with_ops=True)