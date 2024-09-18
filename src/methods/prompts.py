
get_entities_prompt = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": '''
Given text, extract the following metadata according to the format instructions below.
Extract exactly as it is in the text. If it's is more then one separate by comma in list.
 
<< FORMATTING >>
{format_instructions}

<< INPUT >>
{chat_history}
 
<< OUTPUT (remember to include the ```json)>>''',


    "llama3:70b-instruct-q4": '''|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Given text, extract the following metadata according to the format instructions below.
            Extract exactly as it is in the text. If it's is more then one separate by comma in list.
             
            << FORMATTING >>
            {format_instructions}
            
            << INPUT >>
            {chat_history}
             
            << OUTPUT (remember to include the ```json)>>
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

}