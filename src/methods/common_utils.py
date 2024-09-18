from langchain.prompts import PromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

def get_response_schema(labels):

    labels_schema = {
        "person": ResponseSchema(name="person",
                                   description=f"Extract all names of the person in the array format, empty array if none"),
        "country": ResponseSchema(name="country",
                                   description=f"Extract countries kingdoms in the array format, empty array if none"),
        "writer": ResponseSchema(name="writer",
                                   description=f"Extract all names of the person who are writer in the array format, empty array if none"),
        "book": ResponseSchema(name="book",
                                   description=f"Extract names of all the books in the array format, empty array if none"),
        "award": ResponseSchema(name="award",
                                   description=f"Extract names of all the awards in the array format, empty array if none"),
        "literary genre": ResponseSchema(name="literary genre",
                                   description=f"Extract names of all the literary genre in the array format, empty array if none"),
        "poem": ResponseSchema(name="poem",
                                   description=f"Extract names of all the poem in the array format, empty array if none"),
        "location": ResponseSchema(name="location",
                                   description=f"Extract names of all the location in the array format, empty array if none"),
        "event": ResponseSchema(name="event",
                                   description=f"Extract names of all the events in the array format, empty array if none"),
        "organization": ResponseSchema(name="organization",
                                   description=f"Extract names of all the organization in the array format, empty array if none"),
        "magazine": ResponseSchema(name="organization",
                                   description=f"Extract names of all the magazine in the array format, empty array if none"),
        "else": ResponseSchema(name="else",
                                   description=f"Extract all name entity which do not fit any of above d in the array format, empty array if none")
    }

    return [labels_schema[l] for l in labels]