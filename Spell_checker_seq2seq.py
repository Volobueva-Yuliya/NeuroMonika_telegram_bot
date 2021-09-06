# -*- coding: utf-8 -*-
import os
import time
import re

import pandas as pd
import numpy as np
import tensorflow as tf

from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors

print(tf.__version__)

!pip3 install tensorflow==1.13.0rc1

pip install tensorflow-gpu==1.13.1

from google.colab import drive
drive.mount('/content/drive')

def load_book(path):
    """Load a book from its file"""
    input_file = os.path.join(path)
    with open(input_file) as f:
        book = f.read()
    return book

# Считывание имен файлов
path = './books/'
book_files = [f for f in listdir(path) if isfile(join(path, f))]
book_files = book_files[1:]

# Загрузка книг, используя имена файлов
books = []
for book in book_files:
    books.append(load_book(path+book))

# Анализ количества слов в книгах/словарях
for i in range(len(books)):
    print("There are {} words in {}.".format(len(books[i].split()), book_files[i]))

len(books)

# Проверка считанного текста
books[0][:500]

# Удаление ненужных символов и лишних пробелов из текста
def clean_text(text):
    text = re.sub(r'\n', ' . ', text) 
    text = re.sub('\s',' . ', text)
    text = re.sub('[^А-Яа-я\. ]', ' ', text)
    text = re.sub(' \.',' ', text)
    text = re.sub(' +',' ', text)
    return text

clean_books = []
for book in books:
    clean_books.append(clean_text(book))

len(clean_books)

# Проверка очищенного текста
clean_books[0][:500]

# Создание словаря для преобразования словаря (символов) в целые числа
vocab_to_int = {}
count = 0
for book in clean_books:
    for character in book:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1
# Добавление маркеров
codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1

# Проверка словаря
vocab_size = len(vocab_to_int)
print("The vocabulary contains {} characters.".format(vocab_size))
print(sorted(vocab_to_int))

vocab_to_int['а']

# Создание другой словарь для преобразования целых чисел в соответствующие им символы
int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character

# Split the text from the books into sentences.
sentences = []
for book in clean_books:
    for sentence in book.split('. '):
        sentences.append(sentence + '.')
print("There are {} sentences.".format(len(sentences)))

# Проверка корректности разделенного текста по предложениям
sentences[:30]

# Преобразование предложений в целые числа
int_sentences = []

for sentence in sentences:
    int_sentence = []
    for character in sentence:
        int_sentence.append(vocab_to_int[character])
    int_sentences.append(int_sentence)

# Длина каждого предложения
lengths = []
for sentence in int_sentences:
    lengths.append(len(sentence))
lengths = pd.DataFrame(lengths, columns=["counts"])

# Диапазон длины предложений
max_length = 100
min_length = 7

good_sentences = []

for sentence in int_sentences:
    if len(sentence) <= max_length and len(sentence) >= min_length:
        good_sentences.append(sentence)

print("We will use {} to train and test our model.".format(len(good_sentences)))

# Разделение данных на предложения для обучения и тестирования
training, testing = train_test_split(good_sentences, test_size = 0.15, random_state = 2)

print("Number of training sentences:", len(training))
print("Number of testing sentences:", len(testing))

# Сортировка предложений по длине, чтобы уменьшить отступы
# Это позволит модели быстрее обучаться
training_sorted = []
testing_sorted = []


for i in range(min_length, max_length+1):
  for sentence in training:
    if len(sentence) == i:
      training_sorted.append(sentence)
  for sentence in testing:
    if len(sentence) == i:
      testing_sorted.append(sentence)

# Проверка результатов сортировки предложений
for i in range(5):
    print(training_sorted[i], len(training_sorted[i]))

# Функция аугментации
# Перемещение, удаление или добавление символов для создания орфографических ошибок
letters = ['а','б','в','г','д','е','ж','з','и','й','к','л',
           'м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ь','ъ','ы','э','ю','я',]

def noise_maker(sentence, threshold): 
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Большинство символов будут правильными, так как пороговое значение велико
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% случайные символы поменяются местами
            if new_random > 0.67:
                if i == (len(sentence) - 1):
                    # Если последний символ в предложении, он не будет набран
                    continue
                else:
                    # если есть какой-либо другой символ, поменяется порядок на следующий символ
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% в предложение будет добавлена дополнительная строчная буква
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            # ~33% символ не будет напечатан
            else:
                pass     
        i += 1
    return noisy_sentence

# Проверка работы функции аугментации
threshold = 0.95
for sentence in training_sorted[:5]:
    print(sentence)
    print(noise_maker(sentence, threshold))
    print()

# Создайте заполнителей для входных данных в модель
def model_inputs():  
    with tf.name_scope('inputs'):
        inputs = tf.compat.v1.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.compat.v1.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.compat.v1.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.compat.v1.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length

# Удаление идентификатора последнего слова из каждой партии и соединение <GO> с началом каждой партии
def process_encoding_input(targets, vocab_to_int, batch_size):   
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input

# encodung layer
def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):    
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.compat.v1.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop, 
                                                              rnn_inputs,
                                                              sequence_length,
                                                              dtype=tf.float32)

            return enc_output, enc_state
        
        
    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.compat.v1.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, 
                                                            input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, 
                                                            input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                                            cell_bw, 
                                                                            rnn_inputs,
                                                                            sequence_length,
                                                                            dtype=tf.float32)
            # Объединение выходов, так как мы используется двунаправленная RNN
            enc_output = tf.concat(enc_output,2)
            # Используется только прямое состояние, потому что модель не может использовать оба состояния одновременно
            return enc_output, enc_state[0]

# training logits
def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_target_length):
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer) 

        training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_target_length)
        return training_logits

# inference logits (вывод)
def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,
                             max_target_length, batch_size):    
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)

        inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                output_time_major=False,
                                                                impute_finished=True,
                                                                maximum_iterations=max_target_length)


        return inference_logits

## decoding layer and attention
def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, 
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):  
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.compat.v1.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm, 
                                                         input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,
                                                  enc_output,
                                                  inputs_length,
                                                  normalize=False,
                                                  name='BahdanauAttention')
    
    with tf.name_scope("Attention_Wrapper"):
      dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                              attn_mech,
                                                              rnn_size)
    
    #initial_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(enc_state,
                                                                    # _zero_state_tensors(rnn_size, 
                                                                    #                     batch_size, 
                                                                    #                     tf.float32))
    initial_state = dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state)


    with tf.compat.v1.variable_scope("decode"):
       training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                                            sequence_length=targets_length,
                                                            time_major=False)
       training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                           training_helper,
                                                           initial_state,
                                                           output_layer)
      
        # training_logits = training_decoding_layer(dec_embed_input, 
        #                                           targets_length, 
        #                                           dec_cell, 
        #                                           initial_state,
        #                                           output_layer,
        #                                           vocab_size, 
        #                                           max_target_length)


       training_logits, _ ,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,
                                                                    output_time_major=False,
                                                                    impute_finished=True,
                                                                    maximum_iterations=max_target_length)


    with tf.compat.v1.variable_scope("decode", reuse=True):
        # inference_logits = inference_decoding_layer(embeddings,  
        #                                             vocab_to_int['<GO>'], 
        #                                             vocab_to_int['<EOS>'],
        #                                             dec_cell, 
        #                                             initial_state, 
        #                                             output_layer,
        #                                             max_target_length,
        #                                             batch_size)
        start_tokens = tf.tile(tf.constant([vocab_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')
        end_token = (tf.constant(vocab_to_int['<EOS>'], dtype=tf.int32))
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,
                                                                    start_tokens,
                                                                    end_token)
        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                            inference_helper,
                                                            initial_state,
                                                            output_layer)
        inference_logits, _, _= tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                               output_time_major=False,
                                                               impute_finished=True,
                                                               maximum_iterations=max_target_length)
    return training_logits, inference_logits

#seq2seq
def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):  
    enc_embeddings = tf.Variable(tf.random.uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers, 
                                           enc_embed_input, keep_prob, direction)
    
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input, 
                                                        dec_embeddings,
                                                        enc_output,
                                                        enc_state, 
                                                        vocab_size, 
                                                        inputs_length, 
                                                        targets_length, 
                                                        max_target_length,
                                                        rnn_size, 
                                                        vocab_to_int, 
                                                        keep_prob, 
                                                        batch_size,
                                                        num_layers,
                                                        direction)
    
    return training_logits, inference_logits

# предложения дополняются <PAD>, чтобы каждое предложение пакета имело одинаковую длину
def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

# Пакетные предложения, предложения c ошибками и длина предложений
# С каждой эпохой в предложениях будут появляться новые ошибки
def get_batches(sentences, batch_size, threshold):    
    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]
        
        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold))
            
        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)
            
        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))
        
        # Длины для параметров _lengths 
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        
        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))
        
        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths

# Параметры по умолчанию
epochs = 100
batch_size = 128
num_layers = 2
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75

def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):

    tf.compat.v1.reset_default_graph()
    # Входные данные модели    
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    # training and inference logits
    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),
                                                      targets, 
                                                      keep_prob,   
                                                      inputs_length,
                                                      targets_length,
                                                      max_target_length,
                                                      len(vocab_to_int)+1,
                                                      rnn_size, 
                                                      num_layers, 
                                                      vocab_to_int,
                                                      batch_size,
                                                      embedding_size,
                                                      direction)

    # tensors for the training logits and inference logits
    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    # weights for sequence_loss
    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("cost"):
        # Функция потерь
        cost = tf.contrib.seq2seq.sequence_loss(training_logits, 
                                                targets, 
                                                masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # Оптимизатор Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    # Объединение сводок
    merged = tf.summary.merge_all()    

    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph

cd /content/drive/MyDrive/Elbrus/My_works/Final_Project/Data/Spell_Checker/cheki/

# Обучение RNN
def train(model, epochs, log_string):   
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Используется для определения того, когда следует рано прекратить тренировку
        testing_loss_summary = []

        # Какая итерация пакета проходит обучение
        iteration = 0
        
        display_step = 30 #Ход обучения будет отображаться после каждых 30 партий
        stop_early = 0 
        stop = 3 # Если batch_loss_testing не уменьшается при 3 последовательных проверках, обучение прекращается
        per_epoch = 3 # Тестируется модель 3 раза за эпоху
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))

        for epoch_i in range(1, epochs+1): 
            batch_loss = 0
            batch_time = 0
            
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged,
                                             model.cost, 
                                             model.train_op], 
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: keep_probability})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Регистрируется ход обучения
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(training_sorted) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(testing_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost], 
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Запись ходу тестирования
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing, 
                                  batch_time_testing))
                    
                    batch_time_testing = 0

                    # Если batch_loss_testing находится на новом минимуме, модель сохраняется
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!') 
                        stop_early = 0
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break

tf.compat.v1.disable_eager_execution()

# Проверка подключенного оборудования
from tensorflow.python.client import device_lib
def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  print(local_device_protos)
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

get_available_gpus()

# Обучение модели требуемым параметрам настройки
# Загрузка весов
checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"
with tf.Session() as sess:
    # Загрузка модели
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)


for keep_probability in [0.75]:
    for num_layers in [2]:
        for threshold in [0.95]:

            log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                    num_layers,
                                                    threshold) 
            model = build_graph(keep_probability, rnn_size, num_layers, batch_size, 
                                learning_rate, embedding_size, direction)
            train(model, epochs, log_string)

def text_to_ints(text):
    '''Prepare the text for the model'''
    
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

# Проверка правописания
text = 'молок'
text = text_to_ints(text)
checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"
model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction) 

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size, 
                                                 model.inputs_length: [len(text)]*batch_size,
                                                 model.targets_length: [len(text)+1], 
                                                 model.keep_prob: [1.0]})[0]

# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"] 

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))

# Проверка правописания
text = 'тоаврищ'
text = text_to_ints(text)
checkpoint = "./kp=0.75,nl=2,th=0.95.ckpt"
model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction) 

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    
    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size, 
                                                 model.inputs_length: [len(text)]*batch_size,
                                                 model.targets_length: [len(text)+1], 
                                                 model.keep_prob: [1.0]})[0]

# Remove the padding from the generated sentence
pad = vocab_to_int["<PAD>"] 

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
