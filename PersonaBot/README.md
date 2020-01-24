### About
This is a chat API built on openAI's gpt model . It can chat based on predefined personality . can answer basic questions .

### How to use

#### Using Python
```python
# Define url
url = "http://chatbot-url/"
```
##### Home page (aka instruction page)
```python
response = requests.get(url) 
print(response.text) 
```
##### to chat 
```python
url = url+'chat'
params ={'query': 'Hi'}
response = requests.get(url, params)
print(response.json()) 
```
##### get history 
```python
url = url+'history' 
response = requests.get(url) 
print(response.json())
```
##### Get List of available personalities 
```python
url = url+'persona' 
response = requests.get(url) 
print(response.json())
```
##### select personality 
```python
url = url+'set_persona/0' 
response = requests.put(url) 
print(response.text) 
```
##### Add persona 
```python
url = url+'add_persona' 
data ={"persona":"descrive your personality", 
    "back_story":"give a backstory for candidate", 
    "history":"give predefined history" 
    } 

headers = {'Content-type': 'application/json', 'Accept': 'text/plain'} 
response = requests.post(url, data=json.dumps(data), headers=headers) 
print(response.json()) 
```
##### Delete persona 
```python
url = url+'remove_persona/1' 
response = requests.delete(url) 
print(response.json()) 
```
