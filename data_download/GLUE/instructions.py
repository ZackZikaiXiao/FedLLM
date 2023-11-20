INSTRUCTIONS = {
    "acceptability": [
        "Would a linguist rate this sentence to be acceptable linguistically? options: acceptable, unacceptable",
        "How would you consider the linguistic integrity of the preceding sentence? options: acceptable, unacceptable",
        # "Is this test sentence a correct grammatical English sentence?",
        "Is the following sentence linguistically acceptable? options: acceptable, unacceptable",
        # "Would the following sentence, by the strictest standards, be considered correct by a linguist?",
        "Is the next sentence syntactically and semantically acceptable? options: acceptable, unacceptable",
        # "Would a linguist find the following sentence to be a valid English sentence grammatically?",
    ],
    "sentiment": [
        "Is this movie review sentence negative or positive? options: positive, negative",
        "Did the critic thinking positively or negatively of the movie? options: positive, negative",
        "Was the movie seen positively or negatively based on the preceding review? options: positive, negative",
        "How would the sentiment of this sentence be perceived? options: positive, negative",
        "Is the sentiment of the following sentence positive or negative? options: positive, negative",
        "What is the sentiment of the following movie review sentence? options: positive, negative",
        "Would the following phrase be considered positive or negative? options: positive, negative",
        "Does the following review have a positive or negative opinion of the movie? options: positive, negative"
    ],
    "paraphrase_qpp": [
        "Would you say that these questions are the same? options: yes, no",
        "Do those questions have the same meaning? options: yes, no",
        "Are these two questions inquiring about the same information? options: yes, no",
        "Are these two questions paraphrases of each other? options: yes, no",
        "Are these two questions asking the same thing? options: yes, no",
        "Do these questions have the same meaning? options: yes, no",
        "Are the following two questions the same? options: yes, no"
    ],
    "paraphrase_mrpc": [
        "Do they have the same meaning?",
        "Are the two sentences saying the same thing?",
        "Do the above sentences mean the same thing?",
        "Please tell me if the sentences above mean the same.",
        "Are these sentences conveying the same meaning?",
        "Are these two sentences paraphrases of each other?",
        "If the first sentence is true, is the second one also true?",
    ],
    "sentence_similarity": [
        "Rate the textual similarity of these two sentences on a scale from 0 to 5, where 0 is \"no meaning overlap\" and 5 is \"means the same thing\".",
        "On a scale from 0 to 5, where 0 is \"no meaning overlap\" and 5 is \"means the same thing\", how closely does the first sentence resemble the second one?",
        "From 0 to 5 (0=\"no meaning overlap\" and 5=\"means the same thing\"), how similar are the two sentences?",
        "How similar are the following two sentences? Give the answer on a scale from 0 - 5, where 0 is \"not similar at all\" and 5 is \"means the same thing\".",
        "Do the following sentences say the same thing? Return your answer on a scale from 0 to 5, where 0 is \"not similar\" and 5 is \"very similar\".",
        "Rate the similarity of the following two sentences on a scale from 0 to 5, where 0 is \"no meaning overlap\" and 5 is \"means the same thing\".",
        "On a scale from 0-5, where 0 is \"not similar\" and 5 is \"very similar\", how similar are these two sentences?",
    ],
    "MNLI": [
        "Is this second sentence entailed by the first sentence? options: entailment, contradiction, neutral",
        "Does the premise entail the hypothesis? options: entailment, contradiction, neutral",
        "Is the hypothesis entailed by the premise? options: entailment, contradiction, neutral",
        "Is this second sentence entailed by the first sentence? options: entailment, contradiction, neutral",
    ],
    "QNLI": [
        "Does the sentence  answer the question? options: yes, no",
        "Does the sentence  provide a valid answer to the question? options: yes, no",
        "Is the sentence a good answer to the question? options: yes, no",
        "Does the sentence correctly answer the question? options: yes, no",
        "Is the question answered in a satisfactory fashion? options: yes, no",
        "Is the sentence the correct answer? options: yes, no"
    ],
    "RTE": [
        "Based on the premise below can we conclude the hypothesis? options: yes, no",
        "Based on premise can we conclude that hypothesis is true? options: yes, no",
        "Determine if the hypothesis is true based on the premise below. options: yes, no",
        "Can we draw the following hypothesis from the context? options: yes, no",
        "Read the text and determine if the hypothesis is true. options: yes, no",
        "Read the following paragraph and determine if the hypothesis is true. options: yes, no"
    ],
    "WNLI": [
        "Based on the premise below can we conclude the hypothesis? options: yes, no",
        "Based on premise can we conclude that hypothesis is true? options: yes, no",
        "Determine if the hypothesis is true based on the premise below. options: yes, no",
        "Can we draw the following hypothesis from the context? options: yes, no",
        "Read the text and determine if the hypothesis is true. options: yes, no",
        "Read the following paragraph and determine if the hypothesis is true. options: yes, no"
    ],
    "MRPC": [
        "Here are two sentences, do they have the same meaning? options: yes, no",
        "Here are two sentences, are the two sentences saying the same thing? options: yes, no",
        "Do the below sentences mean the same thing? options: yes, no",
        "Please tell me if the sentences above mean the same. options: yes, no",
        "Are these sentences conveying the same meaning? options: yes, no",
        "If the first sentence is true, is the second one also true? options: yes, no",
        "Are these two sentences paraphrases of each other? options: yes, no"
    ]
}
