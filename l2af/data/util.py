import json
from typing import *


def load_dict(fname: str) -> Dict:
    """
    Loading a dictionary from a json file

    :param fname:
    :return:
    """
    with open(fname, 'r') as fp:
        data = json.load(fp)
    return data


def process_ctx_quac(ctx: str) -> str:
    """
    process the context passage of QuAC for ease of use

    :param ctx:
    :return:
    """
    ctx = ctx[:-13]
    ctx = "CANNOTANSWER " + ctx
    return ctx


def process_para_json(paragraph_json: Dict) -> Dict:
    """
    Process a paragraph json

    :param paragraph_json:
    :return:
    """
    paragraph_json["context"] = process_ctx_quac(paragraph_json["context"])
    for qa in paragraph_json["qas"]:
        for answer in qa["prev_ans"]:
            if answer["text"] == "CANNOTANSWER":
                answer["answer_start"] = 0
            else:
                answer["answer_start"] += 13
    return paragraph_json
