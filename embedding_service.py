# coding:utf-8
import tensorflow as tf
import tornado.web
from tornado.options import define, options
import json
import time
import datetime
import numpy as np
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class EmbeddingHandler(tornado.web.RequestHandler):
    def post(self, *args, **kwargs):
        print(self.request.body)
        raw = json.loads(self.request.body.decode('utf-8'),encoding='utf-8')
        query = raw['query']
        embedding = elmo.get_embedding(query)
        result = {'embedding':embedding}
        self.write(json.dumps(result, ensure_ascii=False,cls=MyEncoder))
        self.finish()

class Embedding:
    def __init__(self,vocab_file,options_file,weight_file,token_embedding_file):

        self.vocab_file = vocab_file
        self.options_file = options_file
        self.weight_file = weight_file
        self.token_embedding_file = token_embedding_file

        # Create a TokenBatcher to map text to token ids.
        self.batcher = TokenBatcher(self.vocab_file)

        # Input placeholders to the biLM.
        self.context_token_ids = tf.placeholder('int32', shape=(None, None))

        # Build the biLM graph.
        self.bilm = BidirectionalLanguageModel(
            self.options_file,
            self.weight_file,
            use_character_inputs=False,
            embedding_weight_file=self.token_embedding_file
        )

        # Get ops to compute the LM embeddings.
        self.context_embeddings_op = self.bilm(self.context_token_ids)


        self.elmo_context_input = weight_layers('input', self.context_embeddings_op, l2_coef=0.0)


        self.elmo_context_output = weight_layers(
            'output', self.context_embeddings_op, l2_coef=0.0
        )

    def get_embedding(self,tokenized_context):
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
            # It is necessary to initialize variables once before running inference.
            sess.run(tf.global_variables_initializer())

            # Create batches of data.
            context_ids = self.batcher.batch_sentences(tokenized_context)

            # Compute ELMo representations (here for the input only, for simplicity).
            elmo_context_input_ = sess.run(
                self.elmo_context_input['weighted_op'],
                feed_dict={self.context_token_ids: context_ids}
            )
            return elmo_context_input_

if __name__ == '__main__':
    define("port", default=9094, help="run on the given port", type=int)
    accout = 'nlp'
    vocab_file = '/home/' + accout + '/pySpace/bilm-tf/vocab_bilm.txt'
    options_file = '/home/' + accout + '/pySpace/bilm-tf/output/options.json'
    weight_file = '/home/' + accout + '/pySpace/bilm-tf/output/weights.hdf5'
    token_embedding_file = '/home/' + accout + '/pySpace/bilm-tf/vocab_embedding.hdf5'
    start = time.time()
    elmo = Embedding(vocab_file,options_file,weight_file,token_embedding_file)
    print('实例化完成,用时:',datetime.timedelta(seconds=int(time.time()-start)))
    # 启动tornado http服务
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[(r'/embedding', EmbeddingHandler)],autoreload=False,
        debug=False
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
