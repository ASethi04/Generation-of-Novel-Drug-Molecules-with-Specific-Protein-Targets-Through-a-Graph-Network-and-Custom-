from model import CVAE
from utility_funcs import *
import numpy as np
import os
import tensorflow as tf
import time


#convert smiles to numpy array
molecules_input, molecules_output, char, vocab, labels, length = load_data(args.prop_file, args.seq_length)
vocab_size = len(char)

#make save_dir
if not os.path.isdir(args.save_dir):
    os.mkdir(args.save_dir)

#divide data into training and test set
num_train_data = int(len(molecules_input)*0.75)
train_molecules_input = molecules_input[0:num_train_data]
test_molecules_input = molecules_input[num_train_data:-1]

train_molecules_output = molecules_output[0:num_train_data]
test_molecules_output = molecules_output[num_train_data:-1]

train_labels = labels[0:num_train_data]
test_labels = labels[num_train_data:-1]

train_length = length[0:num_train_data]
test_length = length[num_train_data:-1]

model = CVAE(vocab_size,
             args
             )
print ('Number of parameters : ', np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))

for epoch in range(args.num_epochs):

    st = time.time()
    train_loss = []
    test_loss = []
    st = time.time()
    
    for iteration in range(len(train_molecules_input)//args.batch_size):
        n = np.random.randint(len(train_molecules_input), size = args.batch_size)
        x = np.array([train_molecules_input[i] for i in n])
        y = np.array([train_molecules_output[i] for i in n])
        l = np.array([train_length[i] for i in n])
        c = np.array([train_labels[i] for i in n])
        cost = model.train(x, y, l, c)
        train_loss.append(cost)
    
    for iteration in range(len(test_molecules_input)//args.batch_size):
        n = np.random.randint(len(test_molecules_input), size = args.batch_size)
        x = np.array([test_molecules_input[i] for i in n])
        y = np.array([test_molecules_output[i] for i in n])
        l = np.array([test_length[i] for i in n])
        c = np.array([test_labels[i] for i in n])
        cost = model.test(x, y, l, c)
        test_loss.append(cost)
    
    train_loss = np.mean(np.array(train_loss))        
    test_loss = np.mean(np.array(test_loss))    
    end = time.time()    
    if epoch==0:
        print ('epoch\ttrain_loss\ttest_loss\ttime (s)')
    print ("%s\t%.3f\t%.3f\t%.3f" %(epoch, train_loss, test_loss, end-st))
    ckpt_path = args.save_dir+'/model_'+str(epoch)+'.ckpt'
    model.save(ckpt_path, epoch)
