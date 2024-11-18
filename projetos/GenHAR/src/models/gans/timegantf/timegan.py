import tensorflow as tf
import numpy as np
from models.gans.timegantf.utils import extract_time, rnn_cell, random_generator, batch_generator

class TimeGAN(tf.keras.Model):
    def __init__(self, hidden_dim, num_layers, module_name, dim):
        super(TimeGAN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.module_name = module_name
        self.dim = dim
        
        self.embedder = self.build_rnn("embedder")
        self.recovery = self.build_rnn("recovery", output_dim=dim)
        self.generator = self.build_rnn("generator")
        self.supervisor = self.build_rnn("supervisor", num_layers=num_layers - 1)
        self.discriminator = self.build_rnn("discriminator", output_dim=1)

    def build_rnn(self, name, num_layers=None, output_dim=None):
        if num_layers is None:
            num_layers = self.num_layers
        layers = [rnn_cell(self.module_name, self.hidden_dim) for _ in range(num_layers)]
        if output_dim is not None:
            layers.append(tf.keras.layers.Dense(output_dim, activation='sigmoid'))
        return tf.keras.Sequential(layers)

    def call(self, X, Z, T):
        H = self.embedder(X)
        X_tilde = self.recovery(H)
        
        E_hat = self.generator(Z)  # Gerador produz embeddings
        H_hat = self.supervisor(E_hat)  # Supervisor gera sequência
        X_hat = self.recovery(H_hat)  # Reconstrói os dados a partir das representações latentes

        Y_fake = self.discriminator(H_hat)  # Classifica como falso
        Y_real = self.discriminator(H)  # Classifica como real

        return X_tilde, X_hat, Y_fake, Y_real

def timegan(ori_data, parameters):
    # Não é necessário chamar tf.reset_default_graph() em TensorFlow 2.x

    no, seq_len, dim = np.asarray(ori_data).shape
    ori_time, max_seq_len = extract_time(ori_data)

    def MinMaxScaler(data):
        min_val = np.min(np.min(data, axis=0), axis=0)
        data = data - min_val
        max_val = np.max(np.max(data, axis=0), axis=0)
        norm_data = data / (max_val + 1e-7)
        return norm_data, min_val, max_val
    
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    # Inicialização de parâmetros
    hidden_dim = parameters['hidden_dim']
    num_layers = parameters['num_layer']
    iterations = parameters['iterations']
    batch_size = parameters['batch_size']
    module_name = parameters['module']
    
    # Placeholders
    X = tf.keras.Input(shape=(None, dim))
    Z = tf.keras.Input(shape=(None, dim))
    T = tf.keras.Input(shape=(None,), dtype=tf.int32)

    model = TimeGAN(hidden_dim, num_layers, module_name, dim)

    # Forward pass
    X_tilde, X_hat, Y_fake, Y_real = model(X, Z, T)

    # Perdas
    D_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_real), Y_real)
    D_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(Y_fake), Y_fake)
    D_loss = D_loss_real + D_loss_fake

    G_loss_U = tf.losses.sigmoid_cross_entropy(tf.ones_like(Y_fake), Y_fake)
    G_loss_S = tf.losses.mean_squared_error(X, X_hat)
    G_loss = G_loss_U + 100 * G_loss_S

    # Otimizadores
    e_vars = [v for v in model.trainable_variables if v.name.startswith('embedder')]
    r_vars = [v for v in model.trainable_variables if v.name.startswith('recovery')]
    g_vars = [v for v in model.trainable_variables if v.name.startswith('generator')]
    d_vars = [v for v in model.trainable_variables if v.name.startswith('discriminator')]

    E_solver = tf.keras.optimizers.Adam().minimize(G_loss, var_list=e_vars + r_vars)
    D_solver = tf.keras.optimizers.Adam().minimize(D_loss, var_list=d_vars)
    G_solver = tf.keras.optimizers.Adam().minimize(G_loss, var_list=g_vars)

    # Treinamento...


    # Treinamento
    for itt in range(iterations):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        Z_mb = random_generator(batch_size, dim, T_mb, max_seq_len)
        
        sess.run([E_solver, G_solver], feed_dict={X: X_mb, Z: Z_mb, T: T_mb})

        if itt % 1000 == 0:
            d_loss = sess.run(D_loss, feed_dict={X: X_mb, Z: Z_mb, T: T_mb})
            print(f'Step: {itt}, D_loss: {d_loss}')

    # Geração de dados sintéticos
    Z_mb = random_generator(no, dim, ori_time, max_seq_len)
    generated_data_curr = sess.run(X_hat, feed_dict={Z: Z_mb, X: ori_data, T: ori_time})
    
    generated_data = [generated_data_curr[i, :ori_time[i], :] for i in range(no)]
    generated_data = generated_data * max_val + min_val
    
    return generated_data
