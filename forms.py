import torch
from transformers import BertConfig, CONFIG_NAME, BertTokenizer
from transformers import AutoModel, AutoTokenizer
import os,sys
sys.path.append(os.path.dirname('/home/daegon/AES/'))
from document_bert_architectures import DocumentBertCombineWordDocumentLinear, DocumentBertSentenceChunkAttentionLSTM
# from evaluate import evaluation
from encoder import encode_documents
from data import asap_essay_lengths, fix_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os.path


class DocumentBertScoringModel():
    def __init__(self, chunk_model_path,word_doc_model_path,config):
        
        self.bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")   # transformer = 4.7.0 

        # config설정
        # if os.path.exists(self.args['bert_model_path']):
        #     if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
        #         config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
        #     elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
        #         config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
        #     else:
        #         raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        # else:
        #     config = BertConfig.from_pretrained(self.args['bert_model_path'])
            # config는 제외하자.
        self.config = config
        # self.prompt = int(args.prompt[1])
        self.prompt = 2       # p2
        chunk_sizes_str = '90_30_130_10'
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        self.device = 'cuda'
        if "0" != chunk_sizes_str:
            for chunk_size_str in chunk_sizes_str.split("_"):
                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)
        # bert_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        # print("prompt:%d, asap_essay_length:%d" % (self.prompt, asap_essay_lengths[self.prompt]))
        # print("chunk_sizes_str:%s, bert_batch_size_str:%s" % (chunk_sizes_str, bert_batch_size_str))
        

        # 저장된 파라미터 불러오기 => load_model
        self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
            word_doc_model_path,
            config=config)
        self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
            chunk_model_path,
            config=config)
            
     
    def result_point(self, input_sentence, mode_):    # 예제 넣어보기
        # correct_output = None
        # 토크나이징
        document_representations_word_document, document_sequence_lengths_word_document = encode_documents(
            input_sentence, self.bert_tokenizer, max_input_length=512)
        
        document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
        
        for i in range(len(self.chunk_sizes)):
            document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                input_sentence,
                self.bert_tokenizer,
                max_input_length=self.chunk_sizes[i])   # 맥스길이를 chunk size로 설정
            document_representations_chunk_list.append(document_representations_chunk)
            document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)    # 토크나이즈한거 다 리스트에 추가.
        # correct_output = torch.FloatTensor(data[1])     # data[1]에는 정답이 들어있다.

        self.bert_regression_by_word_document.to(device=self.device)
        self.bert_regression_by_chunk.to(device=self.device)

        self.bert_regression_by_word_document.eval()    # eval 모드로 변경
        self.bert_regression_by_chunk.eval()

        with torch.no_grad():           # 기울기 저장 X
            # predictions = torch.empty((document_representations_word_document.shape[0]))
            # 한 문장 삽입
            document_tensors_word_document = document_representations_word_document[0:0+1].to(device=self.device)
            # 토크나이즈 한 것을 모델에 삽입
            predictions_word_document = self.bert_regression_by_word_document(document_tensors_word_document, device=self.device)
            predictions_word_document = torch.squeeze(predictions_word_document)

            predictions_word_chunk_sentence_doc = predictions_word_document
            for chunk_index in range(len(self.chunk_sizes)):
                document_tensors_chunk = document_representations_chunk_list[chunk_index][0:0+1].to(
                    device=self.device)
                predictions_chunk = self.bert_regression_by_chunk(
                    document_tensors_chunk,
                    device=self.device,
                    bert_batch_size=self.bert_batch_sizes[chunk_index]
                )
                predictions_chunk = torch.squeeze(predictions_chunk)
                predictions_word_chunk_sentence_doc = torch.add(predictions_word_chunk_sentence_doc, predictions_chunk)
            # predictions[0] = predictions_word_chunk_sentence_doc
            
            pred_point = float(predictions_word_chunk_sentence_doc)
            # pred_point range : 0~5
            if pred_point < 0:
                pred_point = 0
            elif pred_point > 5:
                pred_point = 5
            pred_point  *= 20
            pred_point = round(pred_point,2)
                
        # if mode_ == 'logical':
        #     print("{} 예측 점수 : {}점".format('논리성',pred_point))
        
        # elif mode_ == 'novelty':
        #     print("{} 예측 점수 : {}점".format('참신성',pred_point))
        
        # elif mode_ == 'persuasive':
        #     print("{} 예측 점수 : {}점".format('설득력',pred_point))
            
        # else:
        #     print("{} 예측 점수 : {}점".format('?',pred_point))
        
        return pred_point
    
