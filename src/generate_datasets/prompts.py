

label_prompts = {

    "mistralai/Mixtral-8x7B-Instruct-v0.1": '''<s>[INST]
            Generate 50 names/examples of the {} in the array format.
            - Examples might have some {} significance
            - Make sure you generate examples. It is not necessary they need to actually exists in real world.
            - Dont add any extra information
            - Make sure it is in an array format
            [/INST]
            ''',

    "llama3:70b-instruct-q4": '''|begin_of_text|><|start_header_id|>system<|end_header_id|>
            Generate 50 names/examples of the {} in the array format.
            - Examples might have some {} significance
            - Make sure you generate examples. It is not necessary they need to actually exists in real world.
            - Dont add any extra information
            - Make sure it is in an array format
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

}








example_prompt = {
    "mistralai/Mixtral-8x7B-Instruct-v0.1": '''<s>[INST]
You are the master in creating data points for evaluating name entity recognition systems.

You goal is to generate text which contains the provided named entities.\
The theme of the generated text is {}.
Some of the examples are:

{}
Instructions:
1. You task is to generate short text that contains ONLY provided named entities.
2. The text must follow the historical literature theme.
3. Be creative and keep the text natural, follow style and tone of example (the same punctuation style, etc).
4. Try to generate text indistinguishable from examples when it comes to style of text (same punctuation style, lengths of sentences, etc)
5. Make sure all named entities are mentioned in the text.

Generate 1 example with following,
{}
[/INST]
Text:
''',

"llama3:70b-instruct-q4": '''|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are the master in creating data points for evaluating name entity recognition systems.

You goal is to generate text which contains the provided named entities.\
The theme of the generated text is {}.
Some of the examples are:

{}
Instructions:
1. You task is to generate short text that contains ONLY provided named entities.
2. The text must follow the historical literature theme.
3. Be creative and keep the text natural, follow style and tone of example (the same punctuation style, etc).
4. Try to generate text indistinguishable from examples when it comes to style of text (same punctuation style, lengths of sentences, etc)
5. Make sure it is just the text and no extra information.
6. Make sure all named entities are mentioned in the text.

Generate 1 example with following,
{}

Text:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''

}