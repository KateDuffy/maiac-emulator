import argparse
from configparser import ConfigParser
import datetime as dt
import json
import numpy as np
import os
import re
import sys
import tensorflow as tf
from tensorflow import keras
import time

from data import himawari
from models import DCCNN, DCResNet, DCVDSR, loss


def filter_training_files_by_year(data_dir, year):
    filenames = os.listdir(data_dir)
    years = [f.split('_')[1] for f in filenames]
    output_files = []
    for i in range(len(filenames)):
        if years[i] == str(year):
            output_files.append(os.path.join(data_dir, filenames[i]))
    return output_files

def train(training_files,
          testing_files,
          params={'tau': 1e-5, 'priorlengthscale': 1e1},
          learning_rate=1e-4,
          save_dir=None,
          model_name=None,
          batch_size=128,
          device='/gpu:1',
          iterations=int(1e6),
          save_step=int(1e3),
          summary_step=int(1e2),
          N=int(1e9)):

    params_path = './default-checkpoint/bayes_opt/%s_best_parameters_Ax.txt' % model_name
    if os.path.exists(params_path):
        with open(params_path) as json_file:
            params = json.load(json_file)
            print('----loaded best parameters----')
    tau, priorlengthscale = params['tau'], params['priorlengthscale']

    if save_dir is None:
        save_dir = './default-checkpoint'
        save_dir = os.path.join(save_dir, "default-%s-tau-%.3E-pls-%s.ckpt" %(model_name, tau, priorlengthscale))
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    em = himawari.EmulatorData()
    train_set = em.make_dataset(training_files, batch_size=batch_size)
    test_set = em.make_dataset(testing_files, batch_size=batch_size)

    # Use CNN
    output_bands= 6
    if model_name == 'DCFC':
        model = DCCNN(layer_sizes=[512]*3 + [output_bands*2 + 1], filter_sizes=[1]*4,
                      output_bands=output_bands, N=N, tau=tau, priorlengthscale=priorlengthscale)
    elif model_name == 'DCCNN':
        model = DCCNN(layer_sizes=[512]*3 + [output_bands*2 + 1], filter_sizes=[3]*4,
                      output_bands=output_bands, N=N, tau=tau, priorlengthscale=priorlengthscale)
    elif model_name == 'DCResNet':
        model = DCResNet(blocks=5, output_bands=output_bands, N=N,
                         tau=tau, priorlengthscale=priorlengthscale)
    elif model_name == 'DCVDSR':
        model = DCVDSR(hidden_layers=[512]*3, output_bands=output_bands,
                       N=N, tau=tau, priorlengthscale=priorlengthscale)

    optimizer = tf.compat.v2.keras.optimizers.Adam(learning_rate)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
    manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print("Restoring from checkpoint {}".format(manager.latest_checkpoint))

    summary_writer = tf.summary.create_file_writer(save_dir + '/log')
    
    with summary_writer.as_default():
        
        for i in range(iterations):
            element = train_set.get_next()
            x_train, y_train, m_train = element['AHI05'], element['AHI12'], element['mask']
            element = test_set.get_next()
            x_test, y_test, m_test =  element['AHI05'], element['AHI12'], element['mask']

            start_time = time.time()
            with tf.GradientTape() as tape:
                loc, logvar, probs, prediction, reg_losses, dropout_probs = model(x_train, training=True)
                train_loss = loss(y_train, m_train, loc, logvar, probs,
                                  reg_losses=reg_losses, is_training=tf.constant(True), step=ckpt.step)
                

            grads = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            duration = time.time() - start_time
            ckpt.step.assign_add(1)


            if int(ckpt.step) % save_step == 0:
                tf.saved_model.save(model, save_dir) # only the latest model for now
                manager.save()

            if int(ckpt.step) % summary_step == 0:
                loc, logvar, probs, prediction, reg_losses, dropout_probs = model(x_test, training=False)
                test_loss = loss(y_test, m_test, loc, logvar, probs, 
                                 reg_losses=reg_losses, is_training=tf.constant(False), step=ckpt.step)          
                print("Step: %d, Examples/sec: %0.5f, Training Loss: %2.4f, Test Loss: %2.4f" %  \
                      (int(ckpt.step), batch_size / duration, train_loss, test_loss))
                print("dropout probabilities: ", dropout_probs)
                
                for i in range(len(dropout_probs)):
                    tf.compat.v2.summary.scalar('concrete/dropout_prob_%s'%i, tf.reduce_mean(dropout_probs[i]), step=int(ckpt.step))
                
                tf.summary.image('input-band0', tf.expand_dims(x_test[:,:,:,0], -1), step=int(ckpt.step))
                tf.summary.image('label-band0', tf.expand_dims(tf.nn.relu(y_test[:,:,:,0]), -1), step=int(ckpt.step))
                tf.summary.image('output-band0', tf.expand_dims(tf.nn.relu(prediction[:,:,:,0]), -1), step=int(ckpt.step))

    test_loss=0
    for i in range(100):
        element = test_set.get_next()
        x_test, y_test, m_test =  element['AHI05'], element['AHI12'], element['mask']
        loc, logvar, probs, prediction, reg_losses = model(x_test, training=False)
        test_loss += loss(y_test, m_test, loc, logvar, probs, reg_losses=reg_losses, step=ckpt.step, is_training=tf.constant(False))
    return test_loss.numpy()/100.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/nobackupp13/kmduffy1/training/', type=str)
    parser.add_argument('--checkpoint_file', type=str, default=None,
                        help='Checkpoint file to continue training. Default=None.')
    parser.add_argument('--device', type=str, default='/gpu:0',
                        help='Which device do training on, /gpu:0, /cpu:0, etc.')
    parser.add_argument('--model_name', type=str, default='DCCNN',
                       help='Select the model to train. Default=DCCNN')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Select the training batch size. Default=128')
    args = parser.parse_args()


    os.environ["CUDA_VISIBLE_DEVICES"] = args.device[-1]

    training_files = filter_training_files_by_year(args.data_dir, 2016)
    testing_files = filter_training_files_by_year(args.data_dir, 2017)

    train(training_files=training_files,
          testing_files=testing_files,
          model_name='DCCNN',
          summary_step=int(1e2),
          save_step=int(1e3),
          batch_size=50)
