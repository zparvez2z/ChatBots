from transformers import BertTokenizer, BertForQuestionAnswering
import torch
from flask import Flask, request, jsonify,render_template, make_response
from flask_restful import reqparse, abort, Api, Resource
import kb

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = ""
text =  kb.KB

application = Flask(__name__)
api = Api(application)


req_parser = reqparse.RequestParser()
req_parser.add_argument('query')

class Index(Resource):    
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'),200,headers) 

class Chat(Resource):    
    def get(self):
        req_args = req_parser.parse_args()   
        question = req_args['query']
                
        if  question =='': 
            print('question should not be empty!')
            return {"replay":'question should not be empty!'}

        question_ids = tokenizer.encode(question, text)
        token_type_ids = [0 if i <= question_ids.index(102) else 1 for i in range(len(question_ids))]
        start_scores, end_scores = model(torch.tensor([question_ids]), token_type_ids=torch.tensor([token_type_ids]))

        all_tokens = tokenizer.convert_ids_to_tokens(question_ids)
        answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
        answer = answer.replace(" ##", "")
        if "[SEP]" in answer:
            print("I can't figure that out.  May be you can try rephrasing your question")
            return {"reply":"I can't figure that out.  May be you can try rephrasing your question"}

        print("answer : {}".format(answer))
        return {"reply":answer} 

api.add_resource(Index , '/')
api.add_resource(Chat , '/chat')

if __name__ == "__main__":    
    application.run(debug=True)
    

        

