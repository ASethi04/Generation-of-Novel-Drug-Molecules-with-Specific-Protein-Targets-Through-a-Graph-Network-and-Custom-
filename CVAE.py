import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow as tf2 
import tensorflow_addons as tfa
import numpy as np
import threading
from utils import *
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA


class CVAE():
    def __init__(self,
                 vocab_size,
                 char,
                 args
                  ):
        self.char = char
        self.vocab_size = vocab_size
        self.batch_size = args.batch_size
        self.latent_size = args.latent_size
        self.lr = tf.Variable(args.lr, trainable=False)
        self.num_prop = args.num_prop
        self.stddev = args.stddev
        self.mean = args.mean
        self.unit_size = args.unit_size
        self.n_rnn_layer = args.n_rnn_layer
        
        self._create_network()


    def _create_network(self):
        self.X = tf.placeholder(tf.int32, [self.batch_size, None])
        self.Y = tf.placeholder(tf.int32, [self.batch_size, None])
        self.C = tf.placeholder(tf.float32, [self.batch_size, self.num_prop])
        self.L = tf.placeholder(tf.int32, [self.batch_size])
        

        
        decoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        encoded_rnn_size = [self.unit_size for i in range(self.n_rnn_layer)]
        
        with tf.variable_scope('decode'):
            decode_cell=[]
            for i in decoded_rnn_size[:]:
                decode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.decoder = tf.nn.rnn_cell.MultiRNNCell(decode_cell)
        
        with tf.variable_scope('encode'):
            encode_cell=[]
            for i in encoded_rnn_size[:]:
                encode_cell.append(tf.nn.rnn_cell.LSTMCell(i))
            self.encoder = tf.nn.rnn_cell.MultiRNNCell(encode_cell)
        
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(self.unit_size)

        self.weights = {}
        self.biases = {}
        self.eps = {
            'eps' : tf.random_normal([self.batch_size, self.latent_size], stddev=self.stddev, mean=self.mean)
        }


        self.weights['softmax'] = tf.get_variable("softmaxw", initializer=tf.random_uniform(shape=[decoded_rnn_size[-1], self.vocab_size], minval = -0.1, maxval = 0.1))       
        
        self.biases['softmax'] =  tf.get_variable("softmaxb", initializer=tf.zeros(shape=[self.vocab_size]))
        self.weights['out_mean'] = tf.get_variable("outmeanw", initializer=tf2.initializers.GlorotUniform(), shape=[self.unit_size, self.latent_size]),
        self.weights['out_log_sigma'] = tf.get_variable("outlogsigmaw", initializer=tf2.initializers.GlorotUniform(), shape=[self.unit_size, self.latent_size]),
        self.biases['out_mean'] = tf.get_variable("outmeanb", initializer=tf2.initializers.GlorotUniform(), shape=[self.latent_size]),
        self.biases['out_log_sigma'] = tf.get_variable("outlogsigmab", initializer=tf2.initializers.GlorotUniform(), shape=[self.latent_size]),

        self.embedding_encode = tf.get_variable(name = 'encode_embedding', shape = [self.latent_size, self.vocab_size], initializer = tf.random_uniform_initializer( minval = -0.1, maxval = 0.1))
        
        self.embedding = tf.nn.embedding_lookup(self.embedding_encode, self.X)

        
        self.latent_vector, self.mean, self.log_sigma = self.encode()
        
        self.decoded, decoded_logits = self.decode(self.latent_vector)
        self.mol_pred = tf.argmax(self.decoded, axis=2)


        weights = tf.sequence_mask(self.L, tf.shape(self.X)[1])
        weights = tf.cast(weights, tf.int32)
        weights = tf.cast(weights, tf.float32)

        self.reconstr_loss = tf.reduce_mean(tfa.seq2seq.sequence_loss(
            logits=decoded_logits, targets=self.Y, weights=weights))
        self.latent_loss = self.cal_latent_loss(self.mean, self.log_sigma)
        self.real_loss =  tf.py_func(self.convert_to_smiles_tf, inp=[self.mol_pred, list(self.char.keys()), self.reconstr_loss, self.latent_loss, self.Y], Tout=tf.float32)
        
       

        # Loss

        self.loss = self.latent_loss + (1.5*self.reconstr_loss) + self.real_loss
        #self.loss = self.reconstr_loss 
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.opt = optimizer.minimize(self.loss)
        
        self.sess = tf.Session()
        
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        
        self.saver = tf.train.Saver(max_to_keep=None)
        #tf.train.start_queue_runners(sess=self.sess)


        print ("Network Ready")

    def encode(self): 
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        C = tf.expand_dims(self.C, 1)
        C = tf.tile(C, [1, tf.shape(X)[1], 1])
        inp = tf.concat([X, C], axis=-1)

        
        _, state = tf.nn.dynamic_rnn(self.encoder, inp, dtype=tf.float32, scope = 'encode', sequence_length = self.L)
        c,h = state[-1]
        self.weights['out_mean'] = tf.reshape(self.weights['out_mean'], [self.unit_size, -1])
        self.weights['out_log_sigma'] = tf.reshape(self.weights['out_log_sigma'], [self.unit_size, -1])
        mean = tf.matmul(h, self.weights['out_mean'])+self.biases['out_mean']
        log_sigma = tf.matmul(h, self.weights['out_log_sigma'])+self.biases['out_log_sigma']
        

        retval = mean+tf.exp(log_sigma/2.0)*self.eps['eps']
        print("latent space shape", tf.shape(retval))
        return retval, mean, log_sigma

    def decode(self, Z):
        seq_length=tf.shape(self.X)[1]
        print("seq length", seq_length)
        new_Z = tf.tile(tf.expand_dims(Z, 1), [1, seq_length, 1])
        C = tf.expand_dims(self.C, 1)
        C = tf.tile(C, [1, tf.shape(self.X)[1], 1])
        X = tf.nn.embedding_lookup(self.embedding_encode, self.X)
        print("X",  tf.shape(X))
        inputs = tf.concat([new_Z, X, C], axis=-1)
        print("inputs", tf.shape(inputs))
        
        self.initial_decoded_state = tuple([tf.nn.rnn_cell.LSTMStateTuple(tf.zeros((self.batch_size, self.unit_size)), tf.zeros((self.batch_size, self.unit_size))) for i in range(self.n_rnn_layer)])
        Y, self.output_decoded_state = tf.nn.dynamic_rnn(self.decoder, inputs, dtype=tf.float32, scope = 'decode', sequence_length = self.L, initial_state=self.initial_decoded_state)

        Y = tf.reshape(Y, [self.batch_size*seq_length, -1])
        Y = tf.matmul(Y, self.weights['softmax'])+self.biases['softmax']
        Y_logits = tf.reshape(Y, [self.batch_size, seq_length, -1])
        Y = tf.nn.softmax(Y_logits)
        print("Y:", Y, "and logits:", Y_logits)
        return Y, Y_logits

    def save(self, ckpt_path, global_step):
        self.saver.save(self.sess, ckpt_path, global_step = global_step)
        #print("model saved to '%s'" % (ckpt_path))

    def assign_lr(self, learning_rate):
        self.sess.run(tf.assign(self.lr, learning_rate ))
    
    def restore(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)

    def get_latent_vector(self, x, c, l):
        return self.sess.run(self.latent_vector, feed_dict={self.X : x, self.C : c, self.L : l})

    def cal_latent_loss(self, mean, log_sigma):
        latent_loss = tf.reduce_mean(-0.5*(1+log_sigma-tf.square(mean)-tf.exp(log_sigma)))
        return latent_loss
    
    def train(self, x, y, l, c):
        _, loss, embedding = self.sess.run([self.opt, self.loss, self.embedding], feed_dict = {self.X :x, self.Y:y, self.L : l, self.C : c})
        return loss, embedding
    
    def test(self, x, y, l, c):
        mol_pred, loss  = self.sess.run([self.mol_pred, self.loss], feed_dict = {self.X :x, self.Y:y, self.L : l, self.C : c})
        return loss

    
    def convert_to_smiles_tf(self, vector, char, re_loss, la_loss, target):
        list_char = char

        
        vector = vector.astype(int)
        errors = []
        new_string = ""

        for molecule in range(len(vector)):
            drug = []
            y = []
            for letter in range(len(vector[molecule])):
                #print("letter", letter, "corresponds to", list_char[letter], "in", list_char)
                drug.append(list_char[vector[molecule][letter]].decode('utf-8'))
                y.append(list_char[target[molecule][letter]].decode('utf-8'))

            drugS = "".join(drug)
            y = "".join(y)
            print("FINAL DRUG", drugS)
            print("ACTUAL DRUG", y)
            index = drugS.find('E')
            if(index == -1):
                index = 186
            real_molecule = Chem.MolFromSmiles(drugS[0:index])
            real_loss = 0.8
            if real_molecule is not None:
                real_loss = 0.0
            errors.append(real_loss)
        print("Reconstruction loss", re_loss)
        print("Latent loss", la_loss)
        print("Real loss", np.mean(errors) + 0.035*self.numIncBrackets('(', ')', drugS[0:index]) + 0.035*self.numWrongRing(drugS[0:index]))
        return np.float32(np.mean(errors) + 0.035*self.numIncBrackets('(', ')', drugS[0:index]) + 0.035*self.numWrongRing(drugS[0:index]))

    def numIncBrackets(self, charS, charE, string):
        openC= 0
        count =0

        for i in range(len(string)):
            if string[i]==charS:
                openC += 1
            elif string[i]==charE:
                openC -= 1
            if(openC<0):
                count += 1
                openC += 1
        return count+openC
    
    def numWrongRing(self, string):
        sum = 0
        for number in range(20):
            count= 0
            for i in range(len(string)):
                if string[i]==number:
                    count += 1
            if(count != 0 and count==1):
                sum += (count%2)
            elif(count != 0 and count > 1):
                sum += (count%2 + (count - 2))
        return sum
    


    def sample(self, latent_vector, c, start_codon, seq_length):
        l = np.ones((self.batch_size)).astype(np.int32)
        x=start_codon
        preds = []
        for i in range(seq_length):
            if i==0:
                x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict = {self.X:x, self.latent_vector:latent_vector, self.L : l, self.C : c})
            else:
                x, state = self.sess.run([self.mol_pred, self.output_decoded_state], feed_dict = {self.X:x, self.latent_vector:latent_vector, self.L : l, self.C : c, self.initial_decoded_state:state})
            preds.append(x)
        return np.concatenate(preds,1).astype(int).squeeze()
