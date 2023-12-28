import xml.etree.ElementTree as ET

import datasets
import json
import random
logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{DBLP:conf/aaai/RogersKDR20,
  author    = {Anna Rogers and
               Olga Kovaleva and
               Matthew Downey and
               Anna Rumshisky},
  title     = {Getting Closer to {AI} Complete Question Answering: {A} Set of Prerequisite
               Real Tasks},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2020, The Thirty-Second Innovative Applications of Artificial Intelligence
               Conference, {IAAI} 2020, The Tenth {AAAI} Symposium on Educational
               Advances in Artificial Intelligence, {EAAI} 2020, New York, NY, USA,
               February 7-12, 2020},
  pages     = {8722--8731},
  publisher = {{AAAI} Press},
  year      = {2020},
  url       = {https://aaai.org/ojs/index.php/AAAI/article/view/6398},
  timestamp = {Thu, 04 Jun 2020 13:18:48 +0200},
  biburl    = {https://dblp.org/rec/conf/aaai/RogersKDR20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """\
QuAIL is a  reading comprehension dataset. \
QuAIL contains 15K multi-choice questions in texts 300-350 tokens \
long 4 domains (news, user stories, fiction, blogs).\
QuAIL is balanced and annotated for question types.\
"""


class QuailConfig(datasets.BuilderConfig):
    """BuilderConfig for QuAIL."""

    def __init__(self, **kwargs):
        """BuilderConfig for QuAIL.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(QuailConfig, self).__init__(**kwargs)


class Quail(datasets.GeneratorBasedBuilder):
    """QuAIL: The Stanford Question Answering Dataset. Version 1.1."""

    _CHALLENGE_SET = "https://raw.githubusercontent.com/text-machine-lab/quail/master/quail_v1.3/xml/randomized/quail_1.3_challenge_randomized.xml"
    _DEV_SET = "https://raw.githubusercontent.com/text-machine-lab/quail/master/quail_v1.3/xml/randomized/quail_1.3_dev_randomized.xml"
    _TRAIN_SET = "https://raw.githubusercontent.com/text-machine-lab/quail/master/quail_v1.3/xml/randomized/quail_1.3_train_randomized.xml"

    BUILDER_CONFIGS = [
        QuailConfig(
            name="quail",
            version=datasets.Version("1.3.0", ""),
            description="Quail dataset 1.3.0",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "context_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "domain": datasets.Value("string"),
                    "metadata": {
                        "author": datasets.Value("string"),
                        "title": datasets.Value("string"),
                        "url": datasets.Value("string"),
                    },
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "question_type": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        datasets.Value("string"),
                    ),
                    "correct_answer_id": datasets.Value("int32"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://text-machine-lab.github.io/blog/2020/quail/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {"train": self._TRAIN_SET, "dev": self._DEV_SET, "challenge": self._CHALLENGE_SET}
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name="challenge", gen_kwargs={"filepath": downloaded_files["challenge"]}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)
        root = ET.parse(filepath).getroot()
        for text_tag in root.iterfind("text"):
            text_id = text_tag.get("id")
            domain = text_tag.get("domain")
            metadata_tag = text_tag.find("metadata")
            author = metadata_tag.find("author").text.strip()
            title = metadata_tag.find("title").text.strip()
            url = metadata_tag.find("url").text.strip()
            text_body = text_tag.find("text_body").text.strip()
            questions_tag = text_tag.find("questions")
            for q_tag in questions_tag.iterfind("q"):
                question_type = q_tag.get("type", None)
                question_text = q_tag.text.strip()
                question_id = q_tag.get("id")
                answers = []
                answer_id = None
                for i, a_tag in enumerate(q_tag.iterfind("a")):
                    if a_tag.get("correct") == "True":
                        answer_id = i
                    answers.append(a_tag.text.strip())

                id_ = f"{text_id}_{question_id}"
                yield id_, {
                    "id": id_,
                    "context_id": text_id,
                    "question_id": question_id,
                    "question_type": question_type,
                    "domain": domain,
                    "metadata": {"author": author, "title": title, "url": url},
                    "context": text_body,
                    "question": question_text,
                    "answers": answers,
                    "correct_answer_id": answer_id,
                }

def generate_original_train_as_json():
    quail = Quail()
    data = []
    for (id, example) in quail._generate_examples(quail._TRAIN_SET):
        instance={}
        instance['question'] = example['question']
        instance['context'] = example['context']
        instance['domain'] = example['domain']
        instance['response'] = example['correct_answer_id']
        instance['answers'] = example['answers']
        data.append(instance)
    with open("./data_download/quail/original_train.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))

def generate_train_from_origin_train(change_label_ratio):
    data_file = open("./data_download/quail/original_train.json", 'r')
    content = data_file.read()
    original_data = json.loads(content)
    random.shuffle(original_data)
    alphabet=['A', 'B', 'C', 'D']
    if change_label_ratio:
        data_amount_for_each_label = [4657, 3260, 1862, 467]
        label_to_be_filled = 0
    correct_answer_id_num_for_each_domain = {
        'fiction': [0, 0, 0, 0],
        'user_stories': [0, 0, 0, 0],
        'blogs': [0, 0, 0, 0],
        'news': [0, 0, 0, 0],
    }
    generated_data = []
    for data in original_data:
        instance = {}
        if change_label_ratio:
            if (data['response'] == label_to_be_filled):
                data_amount_for_each_label[label_to_be_filled] -= 1
            else:
                data['answers'][label_to_be_filled], data['answers'][data['response']] = data['answers'][data['response']], data['answers'][label_to_be_filled]
                data['response'] = label_to_be_filled
                data_amount_for_each_label[label_to_be_filled] -= 1
            if data_amount_for_each_label[label_to_be_filled] == 0 and label_to_be_filled <= 2:
                label_to_be_filled += 1

        correct_answer_id_num_for_each_domain[data['domain']][data['response']] += 1
        instance['instruction'] = data['question']
        instance['context'] = data['context']
        # convert_answer_list = [' ' + str(index) + ':' + answer + ', ' for index, answer in enumerate(example['answers'])]
        for index, item in enumerate(data['answers']):
            if index == 0:
                instance['instruction'] = instance['instruction'] + " options: "+ alphabet[index] + ':' + item
            elif index == 3:
                instance['instruction'] += ", " + alphabet[index] + ':' + item + "."
            else:
                instance['instruction'] += ", " + alphabet[index] + ':' + item
        instance['response'] = alphabet[data['response']]
        instance['category'] = data['domain']
        generated_data.append(instance)
    data_file.close()
    random.shuffle(generated_data)
    with open("./data_download/quail/train.json", 'w') as write_f:
        write_f.write(json.dumps(generated_data, indent=4))

def generate_dev_json_file():
    quail = Quail()
    data = []
    alphabet=['A', 'B', 'C', 'D']
    for (id,example) in quail._generate_examples(quail._DEV_SET):
        instance = {}
        instance['instruction'] = example['question']
        instance['context'] = example['context']
        for index, item in enumerate(example['answers']):
            if index == 0:
                instance['instruction'] = instance['instruction'] + " options: "+ alphabet[index] + ':' + item
            elif index == 3:
                instance['instruction'] += ", " + alphabet[index] + ':' + item + "."
            else:
                instance['instruction'] += ", " + alphabet[index] + ':' + item
        instance['response'] = alphabet[example['correct_answer_id']]
        instance['category'] = example['domain']
        data.append(instance)
    with open("./data_download/quail/dev.json", 'w') as write_f:
        write_f.write(json.dumps(data, indent=4))



if __name__ == "__main__":
    # generate_original_train_as_json()
    generate_train_from_origin_train(True)
    # generate_dev_json_file()