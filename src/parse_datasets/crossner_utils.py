def crossner_ai_labels_and_desc():
    """provides labels and desc for cross ner AI"""
    desc = "The text is in the domain of artificial intelligence"

    labels = {
        "researcher": "The researcher entity. If a person is working on research (including professor, Ph.D. student, researcher in companies, and etc), you should label it as a researcher entity instead of a person entity.",
        "field": "The research field entity, such as machine learning, deep learning, and natural language processing.",
        "task": "The specific task entity in the research field, such as machine translation and object detection.",
        "product": "The product entity that includes the product (e.g., a certain kind of robot like Pepper), system (e.g., facial recognition system), and toolkit (e.g., TensorFlow and PyTorch).",
        "algorithm": "The algorithm entity. It contains algorithms (e.g., decision trees) and models (e.g., CNN and LSTM).",
        "metrics": "The evaluation metrics, such as F1-score.",
        "programming language": "The programming language, such as Java and Python.",
        "conference": "The conference entity. It contains conference and journal entities."
    }

    return desc, labels


def crossner_lit_labels_and_desc():
    """Provides labels and descriptions for cross-domain NER in literature."""
    desc = "The text is in the domain of literature."

    labels = {
        "book": "The book entity.",
        "poem": "The poem entity.",
        "writer": "The writer entity. If a person is working on literature (including writer, novelist, scriptwriter, poet, and etc), you should label it as a writer entity instead of a person entity.",
        "magazine": "The magazine that publishes articles as well as any other literature work.",
        "literary genre": "The literary genre entity, such as novel and science fiction.",
        "award": "The award entity, usually in the field of literature"
    }

    return desc, labels


def crossner_politics_labels_and_desc():
    """Provides labels and descriptions for CrossNER Politics"""
    desc = "The text is in the domain of politics"

    labels = {
        "politician": "The politician entity. If a person entity is a politician, you should label this person as a politician entity instead of a person entity.",
        "election": "The election entity. If an event entity is an election event, you should label it as an election entity instead of an event entity.",
        "political party": "The political party entity. If an organization entity is a political party, you should label it as a political party entity instead of an organization entity.",
    }

    return desc, labels


def crossner_music_labels_and_desc():
    """Provides labels and descriptions for CrossNER Music"""
    desc = "The text is in the domain of music"

    labels = {
        "music genre": "The music genre entity, such as country music, folk music, and jazz.",
        "song": "The song entity.",
        "band": "The band entity. If an organization belongs to a band, you should label it as a band entity instead of an organization entity.",
        "album": "The album entity.",
        "musical artist": "The musical artist entity. If a person is working in the music area (e.g., singer, composer, or songwriter), you should label it as a musical artist entity instead of a person entity.",
        "musical instrument": "The musical instrument entity, such as piano."
    }

    return desc, labels


def crossner_natural_science_labels_and_desc():
    """Provides labels and descriptions for CrossNER Natural Science"""
    desc = "The text is in the domain of natural science, including biology, chemistry, and astrophysics."

    labels = {
        "university": "The university entity.",
        "discipline": "The discipline entity. It contains the areas and subareas of biology, chemistry, and astrophysics, such as quantum chemistry.",
        "theory": "The theory entity. It includes law and theory entities, such as Ptolemaic planetary theories.",
        "award": "The award entity.",
        "scientist": "If a person entity is a scientist, you should label this person as a scientist entity instead of a person entity.",
        "protein": "The protein entity.",
        "enzyme": "Notice that an enzyme is a special type of protein. Hence, if a protein entity is an enzyme, you should label this protein as an enzyme entity instead of a protein entity.",
        "chemical element": "The chemical element entity. This category contains the chemical elements from the periodic table.",
        "chemical compound": "The chemical compound entity. If a chemical compound entity does not belong to protein or enzyme, you should label it as a chemical compound entity.",
        "astronomical object": "The astronomical object entity.",
        "academic journal": "The academic journal entity."
    }

    return desc, labels


def common_labels():
    labels = {
        "person": "The name of a person should be annotated as a person entity. Only choose this if none of the before categories are relevant.",
        "location": "The location entity, including place, bridge, city, county, etc. Only choose this if none of the before categories are relevant.",
        "country": "The country entity. Only choose this if none of the before categories are relevant.",
        "event": "The event entity, which includes festival, war, summit, campaign, etc. Only choose this if none of the before categories are relevant.",
        "organization": "The organization entity. Only choose this if none of the before categories are relevant.",
        "miscellaneous": "An entity needs to be classified as the miscellaneous type if it does not belong to any other category. Only choose this if none of the before categories are relevant.",
        "else": "An entity needs to be classified as the miscellaneous type if it does not belong to any other category. Only choose this if none of the before categories are relevant."
    }

    return labels
