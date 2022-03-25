d = {} 
device = torch.device('cuda')
years = news_train['year'].values 
months = news_train['month'].values 
days = news_train['day'].values 
hours = news_train['hour'].values 
titles = news_train['title'].values 
contents = news_train['content'].values 

m = nn.Softmax(dim=1) 

for i in range(len(years)):
    datestr = str(years[i]) + '/' + str(months[i]) + '/' + str(days[i]) + '/' + str(hours[i]) 
    d[datestr] = [] 
    
for i in tqdm(range(len(years)), position=0, leave=True):
    datestr = str(years[i]) + '/' + str(months[i]) + '/' + str(days[i]) + '/' + str(hours[i]) 
    title = titles[i] 
    content = contents[i] 
    test_model.eval() 
    input_id, attention_mask, token_type_id = kobert_tokenizer(title, content) 
    input_id = torch.tensor(input_id, dtype=int)
    input_id = torch.reshape(input_id, (-1,512)) 
    input_id = input_id.to(device) 
    
    attention_mask = torch.tensor(attention_mask, dtype=int)
    attention_mask = torch.reshape(attention_mask, (-1,512)) 
    attention_mask = attention_mask.to(device) 
    
    token_type_id = torch.tensor(token_type_id, dtype=int)
    token_type_id = torch.reshape(token_type_id, (-1, 512)) 
    token_type_id = token_type_id.to(device) 
    
    with torch.no_grad():
        output = test_model(input_id, 
                            token_type_ids = token_type_id, 
                            attention_mask = attention_mask)  
    logits = output[0] 
    probs = m(logits)
    probs = probs.detach().cpu().numpy().flatten() 
    
    d[datestr].append(probs) 
