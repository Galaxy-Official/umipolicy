import csv
import random
from abc import ABC, abstractmethod
from typing import List

__all__ = [
    "ClipPrompter",
    "VanillaClipPrompter",
]


class BaseClipPrompter(ABC):
    def __init__(self, label_type: str):
        if label_type.lower().startswith("att"):
            self.label_type = "attribute"
        elif label_type.lower().startswith("aff"):
            self.label_type = "affordance"
        elif label_type.lower().startswith("obj"):
            self.label_type = "object"
        else:
            raise ValueError(f"Unknown label type {label_type}")

    def make_prompt_batch(self, attr_list: List[str]) -> List[str]:
        """make prompt for a batch of attributes/affordances/object strings."""
        return [self.make_prompt(attr) for attr in attr_list]

    @abstractmethod
    def make_prompt(self, attr: str) -> str:
        """
        sample one prompt from the prompt pool.
        """

    @abstractmethod
    def make_all_prompts(self, attr: str) -> list[str]:
        """
        return all the possible prompts of the attribute/affordance/object
        """


class ClipPrompter(BaseClipPrompter):
    """Prompter for CLIP training: only generate statements, not question-answer pairs."""

    def __init__(self, label_type: str):
        super().__init__(label_type)

        self.meta_prompt_templates = [
            "This is an image of {obj_desc}.",
            "This is a photo of {obj_desc}.",
            "This is a picture of {obj_desc}.",
            "This is a photograph of {obj_desc}.",
            "This is a snapshot of {obj_desc}.",
            "This is a depiction of {obj_desc}.",
            "This is a representation of {obj_desc}.",
            "This is a visual of {obj_desc}.",
            "This is a portrait of {obj_desc}.",
            "This is a drawing of {obj_desc}.",
            "This is a painting of {obj_desc}.",
            "This is a sketch of {obj_desc}.",
            "This is a screenshot of {obj_desc}.",
            "This is a scan of {obj_desc}.",
            "This is a capture of {obj_desc}.",
            "This is a shot of {obj_desc}.",
            "This is a view of {obj_desc}.",
            "This is an illustration of {obj_desc}.",
            "This is a photograph of {obj_desc}.",
            "This is a photo of {obj_desc}.",
            "This is a picture of {obj_desc}.",
            "This is a snapshot of {obj_desc}.",
            "This is a depiction of {obj_desc}.",
            "This is a representation of {obj_desc}.",
            "This is a visual of {obj_desc}.",
            "This is a portrait of {obj_desc}.",
            "This is a drawing of {obj_desc}.",
            "This is a painting of {obj_desc}.",
            "This is a sketch of {obj_desc}.",
            "This is a screenshot of {obj_desc}.",
            "This is a scan of {obj_desc}.",
            "This is a capture of {obj_desc}.",
            "This is a shot of {obj_desc}.",
            "This is a view of {obj_desc}.",
            "This is an illustration of {obj_desc}.",
        ]

        self.synonym_of_object = [
            "object",
            "item",
            "thing",
            "entity",
            "stuff",
        ]

        if self.label_type == "attribute":
            self.obj_templates = [
                "a {attr} {syn_of_obj}",
                "an {syn_of_obj} with attribute {attr}",
            ]
        elif self.label_type == "affordance":
            self.obj_templates = [
                "a {attr}-able {syn_of_obj}",
                "a {syn_of_obj} that can {attr}",
                "a {syn_of_obj} that is used to {attr}",
                "a {syn_of_obj} that is for {attr}",
                "the function of {syn_of_obj} is {attr}",
                "the affordance of {syn_of_obj} is {attr}",
                "a {syn_of_obj} with function {attr}",
                "a {syn_of_obj} with affordance {attr}",
            ]
        elif self.label_type == "object":
            self.obj_templates = [
                "a {syn_of_obj}",
            ]
        else:
            raise ValueError(f"Unknown label type {label_type}")

    def make_prompt(self, attr: str) -> str:
        """Example: attr = "red"
        syn_of_obj = "object"
        obj_template = "a {attr} {syn_of_obj}"
        meta_template = "This is a photo of {obj_desc}."
        obj_desc = "a red object"
        prompt = "This is a photo of a red object."
        """
        syn_of_obj = random.choice(self.synonym_of_object)
        obj_template = random.choice(self.obj_templates)
        meta_template = random.choice(self.meta_prompt_templates)
        obj_desc = obj_template.format(attr=attr, syn_of_obj=syn_of_obj)
        prompt = meta_template.format(obj_desc=obj_desc)
        return prompt

    def make_all_prompts(self, attr: str) -> List[str]:
        """Example: attr = "red"
        syn_of_obj = "object"
        obj_template = "a {attr} {syn_of_obj}"
        meta_template = "This is a photo of {obj_desc}."
        obj_desc = "a red object"
        prompt = "This is a photo of a red object."
        """
        prompts = []
        for syn_of_obj in self.synonym_of_object:
            for obj_template in self.obj_templates:
                for meta_template in self.meta_prompt_templates:
                    obj_desc = obj_template.format(
                        attr=attr, syn_of_obj=syn_of_obj
                    )
                    prompt = meta_template.format(obj_desc=obj_desc)
                    prompts.append(prompt)
        return prompts


class VanillaClipPrompter(BaseClipPrompter):
    """Prompter for CLIP training: only generate statements, not question-answer pairs."""

    def make_prompt(self, attr: str) -> str:
        if self.label_type in ["attribute", "object"]:
            return f"an object with attribute {attr}"
        else:
            return f"The object that is used to {attr}"
        # if self.label_type in ["attribute", "object"]:
        #     return attr
        # else:
        #     return f"{attr}-able"

    def make_all_prompts(self, attr: str) -> List[str]:
        return [self.make_prompt(attr)]


class CausalPrompter:
    SYNONYM_OF_OBJECTS = [
        "object",
        "item",
        "thing",
        "entity",
        "stuff",
    ]

    META_PROMPT_TEMPLATES = [
        "This is an image of {obj_desc}.",
        "This is a photo of {obj_desc}.",
        "This is a picture of {obj_desc}.",
        "This is a photograph of {obj_desc}.",
        "This is a snapshot of {obj_desc}.",
        "This is a depiction of {obj_desc}.",
        "This is a representation of {obj_desc}.",
        "This is a visual of {obj_desc}.",
        "This is a portrait of {obj_desc}.",
        "This is a drawing of {obj_desc}.",
        "This is a painting of {obj_desc}.",
        "This is a sketch of {obj_desc}.",
        "This is a screenshot of {obj_desc}.",
        "This is a scan of {obj_desc}.",
        "This is a capture of {obj_desc}.",
        "This is a shot of {obj_desc}.",
        "This is a view of {obj_desc}.",
        "This is an illustration of {obj_desc}.",
        "This is a photograph of {obj_desc}.",
        "This is a photo of {obj_desc}.",
        "This is a picture of {obj_desc}.",
        "This is a snapshot of {obj_desc}.",
        "This is a depiction of {obj_desc}.",
        "This is a representation of {obj_desc}.",
        "This is a visual of {obj_desc}.",
        "This is a portrait of {obj_desc}.",
        "This is a drawing of {obj_desc}.",
        "This is a painting of {obj_desc}.",
        "This is a sketch of {obj_desc}.",
        "This is a screenshot of {obj_desc}.",
        "This is a scan of {obj_desc}.",
        "This is a capture of {obj_desc}.",
        "This is a shot of {obj_desc}.",
        "This is a view of {obj_desc}.",
        "This is an illustration of {obj_desc}.",
    ]

    # TODO the object templates is just a draft, I plan to modify it in the future
    OBJECT_TEMPLATES = [
        "If the {syn_of_obj} is {attr}, can we {aff}?",
        "Can we {aff} if the {syn_of_obj} is {attr}?",
        "If the {syn_of_obj} is {attr}, is it possible to {aff}?",
        "Is it possible to {aff} if the {syn_of_obj} is {attr}?",
        "If the {syn_of_obj} is {attr}, will we {aff}?",
        "Will we {aff} if the {syn_of_obj} is {attr}?",
        "If the {syn_of_obj} is {attr}, can we {aff}?",
        "Can we {aff} if the {syn_of_obj} is {attr}?",
        "If the {syn_of_obj} is {attr}, is it possible to {aff}?",
        "Is it possible to {aff} if the {syn_of_obj} is {attr}?",
        "If the {syn_of_obj} is {attr}, will we {aff}?",
        "Will we {aff} if the {syn_of_obj} is {attr}?",
    ]

    def __init__(self):
        super().__init__()

    def question(self, attr: str, aff: str) -> str:
        """
        Returns a short question that references the attribute and affordance.
        Example:
            "If the object is sealed, can we pour out its contents?"
        """
        syn_of_obj = random.choice(self.SYNONYM_OF_OBJECTS)
        obj_template = random.choice(self.OBJECT_TEMPLATES)
        random.choice(self.META_PROMPT_TEMPLATES)
        prompt = obj_template.format(
            syn_of_obj=syn_of_obj,
            attr=attr,
            aff=aff,
        )
        # prompt = meta_template.format(obj_desc=obj_desc)
        return prompt

    def answer(self, is_positive) -> str:
        """
        Returns a short yes/no answer (with brief reasoning).
        """
        if is_positive:
            return "yes"
        else:
            return "no"


class VQAPrompter:

    META_COT_PROMPT_TEMPLATES = [
        "Let's think step by step. The object has the following attributes: {attributes}. ",
        "Based on these attributes, what function can it perform?",
        "Analyzing the attributes {attributes}, which of the following functions is most likely?",
        "First, consider the attributes {attributes}. Then, select the most plausible function.",
    ]

    META_INSTRUCTION = """
    You are an Al assistant to help me matching an answer with several options of a single-choice question. \n
    You are provided with a question, several options, and an answer, \n
    and you need to find which option is most similar to the answer. \n
    If the meaning of all options are significantly different from the answer, output E. \n
    You should output a single uppercase character in A, B, C, D (if they are valid options), and E. \n
    """

    META_PROMPT_TEMPLATES = [
        "This is an image of {obj_desc}.",
        "This is a photo of {obj_desc}.",
        "This is a picture of {obj_desc}.",
        "This is a photograph of {obj_desc}.",
        "This is a snapshot of {obj_desc}.",
        "This is a depiction of {obj_desc}.",
        "This is a representation of {obj_desc}.",
        "This is a visual of {obj_desc}.",
        "This is a portrait of {obj_desc}.",
        "This is a drawing of {obj_desc}.",
        "This is a painting of {obj_desc}.",
        "This is a sketch of {obj_desc}.",
        "This is a screenshot of {obj_desc}.",
        "This is a scan of {obj_desc}.",
        "This is a capture of {obj_desc}.",
        "This is a shot of {obj_desc}.",
        "This is a view of {obj_desc}.",
        "This is an illustration of {obj_desc}.",
    ]

    META_QUESTION_TEMPLATES = [
        "Which attribute best describes the object?",
        "What characteristic is most evident in the object?",
        "Can you identify the feature highlighted by the item?",
        "Which trait is depicted by the object?",
        "What quality is associated with the object?",
        "How would you describe the key attribute of the item?",
        "What specific feature stands out in the object?",
        "What is the defining characteristic of the object?",
        "Which aspect of the object is captured?",
        "What is the notable property of the item?",
        "How is the object best characterized?",
        "What attribute can be observed in the object?",
        "Which feature is most prominent in the object?",
        "What is the distinguishing trait of the object?",
        "What is the key quality of the item?",
        "What aspect defines the object?",
        "Which characteristic is apparent in the object?",
        "What is the prominent feature of the object?",
        "How can the attribute of the object be best described?",
        "Which property is illustrated by the object?",
    ]

    META_COT_QUESTION_TEMPLATES = [
        "What function can this object perform?",
        "Which action is feasible for this object?",
        "What is the most likely use of this object?",
    ]

    # Need attr_temp.csv and aff.csv in VLMEvalKit/scripts
    def __init__(self, attr_file: str, aff_file: str):
        self.attributes = self.load_file(attr_file)
        self.affordances = self.load_file(aff_file)

    def load_file(self, file_path: str) -> List[str]:
        with open(file_path, "r") as file:
            reader = csv.reader(file, delimiter="\t")
            data = [row[0] for row in reader]
        return data

    # TODO extract obj
    def _question(self, attr: list, obj: str, ans_num: int) -> dict:
        """基础问题生成函数"""
        self.obj_desc = obj
        fake_opts = [att for att in self.attributes if att not in attr]
        attr_list = random.sample(attr, ans_num)
        opts = attr_list + random.sample(fake_opts, 4-ans_num)
        random.shuffle(opts)
        ques = {}
        ques["Question"] = (
            random.choice(self.META_PROMPT_TEMPLATES).format(
                obj_desc=self.obj_desc
            )
            + " "
            + random.choice(self.META_QUESTION_TEMPLATES)
        )
        ques.update({i: opt for i, opt in zip("ABCD", opts)})

        ans = [k for k, v in ques.items() if v in attr]
        ques["Answer"] = ans
        return ques, attr_list
    
    def question(self, attr: list, obj: str) -> dict:
        """单选题生成函数"""
        ans_num = 1
        return self._question(attr, obj, ans_num)
    
    def mul_question(self, attr:list, obj:str):
        """不定项选择题生成函数"""
        ans_num = min(len(attr), 4)
        ans_num = random.randint(1, ans_num)
        return self._question(attr, obj, ans_num)

    def CoT_question(self, attr: list, aff: list, obj: str, aff_num: int) -> dict:
        """生成包含思维链的问题"""
        if not aff:
            raise ValueError("aff 列表不能为空")

        self.obj_desc = obj

        correct_aff = random.sample(aff, aff_num)
        fake_aff = [a for a in self.affordances if a != correct_aff]
        fake_samples = random.sample(fake_aff, 4 - aff_num)

        opts = correct_aff + fake_samples
        random.shuffle(opts)
        options = {key: val for key, val in zip("ABCD", opts)}

        cot_prompt = random.choice(self.META_COT_PROMPT_TEMPLATES).format(
            attributes=", ".join(attr)
        )
        question_base = random.choice(self.META_PROMPT_TEMPLATES).format(
            obj_desc=self.obj_desc
        )
        question_full = f"{question_base} {cot_prompt} {random.choice(self.META_COT_QUESTION_TEMPLATES)}"

        answer_key = [k for k, v in options.items() if v == correct_aff][0]

        cot_reasoning = "Attributes {attr} imply the function '{correct_aff}'."
        cot_reasoning.format(
            attr= ", ".join(attr),
            correct_aff= ", ".join(correct_aff)
        )

        return {
            "Question": question_full,
            **options,
            "Answer": answer_key,
            "CoT_Reasoning": cot_reasoning,
        }

    def CoT_cir_question(self, attr: list, aff: list, obj: str) -> List[dict]:
        original_ques = self.CoT_question(attr, aff, obj)
        cir_ques = []

        # 提取选项部分（排除Question/Answer/CoT_Reasoning）
        opts = {k: v for k, v in original_ques.items() if k in "ABCD"}

        for _ in range(4):
            # 打乱选项
            shuffled_items = list(opts.items())
            random.shuffle(shuffled_items)

            # 创建新旧选项键映射
            key_mapping = {
                old[0]: new for new, old in zip("ABCD", shuffled_items)
            }
            # 构建新问题
            new_ques = {}
             # 保留原始问题结构和附加信息
            new_ques.update(
                {
                    "Question": original_ques["Question"],
                    "Answer": key_mapping[original_ques["Answer"]],
                    "CoT_Reasoning": original_ques["CoT_Reasoning"],
                }
            )
            new_ques.update(
                {
                new_key: val
                for (old_key, val), new_key in zip(shuffled_items, "ABCD")
                }
            )

           

            cir_ques.append(new_ques)
        return cir_ques

    def cir_question(self, attr: list, obj: str) -> List[dict]:
        original_ques = self.question(attr, obj)
        return self._generate_cir_variants(original_ques)

    def _generate_cir_variants(self, original_ques: dict) -> List[dict]:
        cir_ques = []
        opts = {k: v for k, v in original_ques.items() if k in "ABCD"}

        for _ in range(4):
            shuffled_items = list(opts.items())
            random.shuffle(shuffled_items)

            key_mapping = {
                old[0]: new for new, old in zip("ABCD", shuffled_items)
            }
            new_ques = {}
            new_ques.update(
                {
                    "Question": original_ques["Question"],
                    "Answer": key_mapping[original_ques["Answer"]],
                }
            )

            new_ques.update(
                {
                new_key: val
                for (old_key, val), new_key in zip(shuffled_items, "ABCD")
                }
            )

            
            # 保留额外字段（如CoT_Reasoning）
            if "CoT_Reasoning" in original_ques:
                new_ques["CoT_Reasoning"] = original_ques["CoT_Reasoning"]

            cir_ques.append(new_ques)
        return cir_ques
