from prompter import VQAPrompter

attr_path = "/home/xiangyushun/workspace/VLMEvalKit/scripts/aff.tsv"
aff_path = "/home/xiangyushun/workspace/VLMEvalKit/scripts/attr.tsv"

prompter = VQAPrompter(attr_path, aff_path)
CoT_questions = prompter.CoT_cir_question(["full", "closed"], ["pour out"], "cup")
# print(CoT_questions)