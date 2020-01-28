import requests,json,re


url = "http://127.0.0.1:5000/"

# test
# response = requests.get(url)
# print(response.text)

# to chat
# url = url+'chat'
# params ={'query': 'Hi'}
# response = requests.get(url, params)
# print(response.json())


# get history
# url = url+'history'
# response = requests.get(url)
# print(response.json())

#Get List of available personalities
# url = url+'persona'
# response = requests.get(url)
# print(response.json())

#select personality
# url = url+'set_persona/0'
# response = requests.put(url)
# print(response.text)


# data_file='dataset_test.json'
# with open(data_file, "r", encoding="utf-8") as f:
#     dataset = json.loads(f.read())

# new_persona={
#             "personality": [
#                 "i like to code .",
#                 "i like python .",
#                 "i like data science !.",
#                 "my favorite holiday is christmas ."
#             ],
#             "utterances": [
#                 {
#                     "candidates": [],
#                     "history": []
#                 }
#             ]
#         }

# print(new_persona)
# dataset["train"].append(new_persona)
# print(json.dumps(dataset["train"],indent=1))


# with open(data_file, "w", encoding="utf-8") as outfile:
#     json.dump(dataset, outfile)


# Add persona
# url = url+'add_persona'
# data ={"persona":"my name is andrew .i like to code . i like python .i like data science . my favorite holiday is christmas.",
#         "back_story":"",
#         "history":"i'm great today ! i'm reading a new book on deep learning"

#         }
        
# headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
# response = requests.post(url, data=json.dumps(data), headers=headers)
# print(response.json())


#Delete persona
# url = url+'remove_persona/1'
# response = requests.delete(url)
# print(response.json())




# # print(json.dumps(dataset["train"][0]["personality"], indent=1))
# persona={}
# for i,data in enumerate(dataset["train"]):
#     persona["persona_id : "+str(i)]=data["personality"]
# print(persona)


# from jsonschema import validate
# persona_schema = {'type':'object',
#                     'properties':{
#                     'persona':{ 'type': 'string' },
#                     'back_story':{ 'type': 'string' },
#                     'history':{ 'type': 'string' }
#                     }
#                 }


# data ={"persona":"my name is andrew .i like to code . i like python .i like data science . my favorite holiday is christmas.",
#         "back_story":" i started programming in 2015 . it is really fun and challenging. i started my programming using c.\
#         now i code in python most of the time.",
#         "history":"i'm great today ! i'm reading a new book on deep learning"

#         }

# validate(data,persona_schema)
# print('ok')

# curl http://localhost:5000/chat?query=hi -v
# curl http://localhost:5000/history -v
# curl http://localhost:5000/persona -v
# curl http://localhost:5000/set_persona/<int:persona_id> -X PUT -v
# curl  -d @data.json -H "Content-Type: application/json" http://localhost:5000/add_persona -X POST -v
# curl http://localhost:5000/remove_persona/<int:persona_id> -X DELETE -v
# curl http://localhost:5000/remove_persona/2 -X DELETE -v
