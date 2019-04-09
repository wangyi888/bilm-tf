import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

# Our small dataset.
tokenized_context = [
    ['被告人','张某某','秘密','窃取','他人财物','价值2210元','××数额较大']
]
#tokenized_context = [sentence.split() for sentence in raw_context]
# tokenized_question = [
#     ['这', '是', '什么'],
# ]

vocab_file='/home/nlp/pySpace/bilm-tf/vocab_bilm.txt'
options_file='/home/nlp/pySpace/bilm-tf/output/options.json'
weight_file='/home/nlp/pySpace/bilm-tf/output/weights.hdf5'
token_embedding_file='/home/nlp/pySpace/bilm-tf/vocab_embedding.hdf5'

## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))
#question_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)
#question_embeddings_op = bilm(question_token_ids)

elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
# with tf.variable_scope('', reuse=True):
#     # the reuse=True scope reuses weights from the context for the question
#     elmo_question_input = weight_layers(
#         'input', question_embeddings_op, l2_coef=0.0
#     )

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
# with tf.variable_scope('', reuse=True):
#     # the reuse=True scope reuses weights from the context for the question
#     elmo_question_output = weight_layers(
#         'output', question_embeddings_op, l2_coef=0.0
#     )

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    #question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_ = sess.run(
        elmo_context_input['weighted_op'],
        feed_dict={context_token_ids: context_ids}
    )

print(elmo_context_input_,'sentence_num:',len(elmo_context_input_),'word_num:',len(elmo_context_input_[0]),'embedding_dim:',len(elmo_context_input_[0][0]))
with open('test_result.txt','w',encoding='utf-8') as out:
    elmo_context_input_ = elmo_context_input_[0]
    for i in range(len(elmo_context_input_)):
        for j in range(len(elmo_context_input_[i])):
            out.write(str(elmo_context_input_[i][j])+',')
            out.flush()

