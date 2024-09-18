# Generation prompt

# intent recognition
generation_intent_prompt = '''You are an AI assistant tasked with generating different short text sample for evaluating intent recognition systems. Your goal is to create a creative text that is representative of the intent provided by the user. Some examples are

{example_string}
Follow these instructions to generate the text:

1. Generate a text with the intent provided by the user.
2. The size of text shoule be similar to the examples provided above
3. Be creative and keep the text natural, following the style and tone of the examples provided above.
4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
5. Make absolutely sure the generated text has the same intent as provided by the user


After generating the text, please provide your output in the following format:

</generated_text>
[Your generated text here]
</generated_text>


Remember, the goal is to create realistic-looking sample that could be used to test intent recognition systems. Make sure your text sounds natural and incorporates the intent provided by the user.'''


# label generation prompt for NER

label_generation_ner = '''You are tasked with generating 50 examples of a specific type of label. Your goal is to create a diverse list that includes both real and fictional examples, some of which may have {theme} significance.

    Guidelines for generating names/examples:
    - Include a mix of real and fictional names
    - Some examples should have potential {theme} significance
    - Ensure diversity in the list (e.g., different cultures, time periods, etc.)
    - Be creative, but keep the names/examples plausible for the given person type
    - Do not add any extra information beyond the names/examples themselves


    Please provide your output as an array of 50 names/examples. Use the following format:

    <output>
    [
      "Example 1",
      "Example 2",
      "Example 3",
      ...
      "Example 50"
    ]
    </output>


    The type of label you should generate names/examples for is:

    <label_type>
    {label_type}
    </label_type>

    - Remember to include entries in the array, separated by commas, and enclosed in square brackets. Do not include any additional information or explanations outside of the array.
    - Make up examples if you don't know about it.
    - Remember to include each example in a double quote

    '''

# NER generation prompt

generation_ner_prompt = '''You are an AI assistant tasked with generating a short text sample for evaluating name entity recognition systems. Your goal is to create a text that incorporates all of the provided named entities provided by the user while maintaining a natural and creative style provided in the example. Some examples are

{example_string}


Follow these instructions to generate the text:

1. Create a short text that incorporates ALL of the provided named entities.
2. Ensure the text follows the theme similar to examples described above.
3. Be creative and keep the text natural, following the style and tone of the examples provided in the original task description.
4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
5. Make absolutely sure that all named entities are mentioned in the text.
6. Don't add quotations or apostrophe.
7. Do not include any other named entities

When writing your text:
- Use proper capitalization and punctuation.
- Vary sentence structure to maintain a natural flow.
- Create logical connections between the named entities to form a coherent narrative.
- Aim for a length and theme similar to the examples above.

After generating the text, please provide your output in the following format:

<generated_text>
[Your generated text here]
</generated_text>


Remember, the goal is to create a realistic-looking sample that could be used to test name entity recognition systems. Make sure your text sounds natural and incorporates all the named entities exactly as given by the user.'''


generation_text_similarity_prompt = f'''You are an AI assistant tasked with generating to short headlines for evaluating text similairty systems.

    The rating scale is summarized by the following guidelines:

        4, Very Similar -- The two items have very similar meanings and the most important ideas, concepts, or actions in the larger text are represented in the smaller text. Some less important information may be missing, but the smaller text is a very good summary of the larger text.
        3, Somewhat Similar -- The two items share many of the same important ideas, concepts, or actions, but include slightly different details. The smaller text may use similar but not identical concepts (e.g., car vs. vehicle), or may omit a few of the more important ideas present in the larger text.
        2, Somewhat related but not similar -- The two items have dissimilar meaning, but shared concepts, ideas, and actions that are related. The smaller text may use related but not necessary similar concepts (window vs. house) but should still share some overlapping concepts, ideas, or actions with the larger text.
        1, Slightly related -- The two items describe dissimilar concepts, ideas and actions, but may share some small details or domain in common and might be likely to be found together in a longer document on the same topic.
        0, Unrelated -- The two items do not mean the same thing and are not on the same topic.

    Your goal is to create two headlines that is representative of the score provided by the user. Some examples are

    {example_string}

    Follow these instructions to generate the text:

    1. Generate a text with the score provided by the user.
    2. The size of text should be similar to the examples provided above following the headline theme.
    3. Be creative and keep the text natural, following the style and tone of the examples provided above.
    4. Try to generate text that is indistinguishable from the examples when it comes to style (use similar punctuation, sentence lengths, etc.).
    5. Make absolutely sure the generated text has the same scoore as provided by the user



    After generating the text, please provide your output in the following format:

    </generated_text>
    T1: The first text goes here
    T2: The second text goes here
    </generated_text>

    Do not generate anything apart from the text, and strictly follow the structure defined above.

    Remember, the goal is to create realistic-looking sample that could be used to test sentence similarity systems. Make sure your text sounds natural and incorporates the theme of the examples above.
    '''


# Benchmarking prompt

benchmarking_intent_prompt = '''Your goal is to extract the intent from the text provided below. The intent can be the following

    {candidate_intents}

    Follow these guidelines when extracting the metadata:
    - Make sure the intent is exactly as it appears in the above list.
    - Don't add any form of reason
    - Make sure the answer is only intent and nothing else


    Provide intent for:

    {intent_text}


    Make sure the generated text is only one of the candidate intent. And do **not** provide any other reasoning
    '''

benchmarking_ner_prompt = '''Your goal is to extract the following metadata from the text:

        {meta_data}

         Follow these guidelines when extracting the metadata:

            - Extract the information exactly as it appears in the text.
            - If multiple items are found for a category, separate them with commas in a list.
            - If no information is found for a category, leave it as an empty array.
            - Use all relevant information you find in the text.


        Present your findings in a JSON format, using a markdown code block. Use the following structure:

            ```json
            {json_format}
            ```



            Please extract all the meta data as specified above for the following text:

            {ner_text}


            - Fill in each array with the appropriate extracted information. If no information is found for a category, leave the array empty.

            - Provide your complete answer in json format as described above. 
            - Please don't add any other information.
            - The text should only mention json once
            {extra_string}

            '''

benchmarking_text_similarity_prompt = '''Your goal is to rate the similarity between the two texts provided by te user.

         The rating scale is summarized by the following guidelines:

        4, Very Similar -- The two items have very similar meanings and the most important ideas, concepts, or actions in the larger text are represented in the smaller text. Some less important information may be missing, but the smaller text is a very good summary of the larger text.
        3, Somewhat Similar -- The two items share many of the same important ideas, concepts, or actions, but include slightly different details. The smaller text may use similar but not identical concepts (e.g., car vs. vehicle), or may omit a few of the more important ideas present in the larger text.
        2, Somewhat related but not similar -- The two items have dissimilar meaning, but shared concepts, ideas, and actions that are related. The smaller text may use related but not necessary similar concepts (window vs. house) but should still share some overlapping concepts, ideas, or actions with the larger text.
        1, Slightly related -- The two items describe dissimilar concepts, ideas and actions, but may share some small details or domain in common and might be likely to be found together in a longer document on the same topic.
        0, Unrelated -- The two items do not mean the same thing and are not on the same topic.



        After rate the similarity between the text T1 and T2, please provide your output in the following format:

        </similarity_rating>
        [Your similarity score goes here]
        </similarity_rating>


        - Do not provide any additional reasoning.

        Provide similarity rating for

        {string_text}
        '''
