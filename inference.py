import json
from vllm import LLM, SamplingParams
from transformers import AutoModel
class Competition:
    def __init__(self):
        #加载获取参数量
        model = AutoModel.from_pretrained(
            "./qwen2.5-3b-merged",
            torch_dtype="auto",
            device_map="auto"
        )
        print(f"Total parameters: {model.num_parameters() / 1e9:.2f}B")
        del model

        #以VLLM加载
        self.llm = LLM(model="./qwen2.5-3b-merged")  #注意相对引用路径
        self.sampling_params = {
            "choice": SamplingParams(max_tokens=1024, temperature=0.8, top_p=0.95),
            "code-generate": SamplingParams(n=3, max_tokens=2048, temperature=0.8, top_p=0.95),
            "generic-generate": SamplingParams(max_tokens=128, temperature=0.8, top_p=0.95),
            "math": SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
        }
        self.prompt_templates = {
            "choice": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}",
            "code-generate": "Read the following function signature and docstring, and fully implement the function described. Your response should only contain the code for this function.\n",
            "generic-generate": "You will be asked to read a passage and answer a question. Think step by step, then write a line of the form 'Answer: $ANSWER' at the end of your response.",
            "math": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \$ANSWER (without quotes) where $ANSWER is the answer to the problem.\n\n{Question}\n\nRemember to put your answer on its own line after 'Answer:', and you do need to use a \\boxed command."
        }
    def load_data(self, data_file="A-data.jsonl"):
        a_datas = []
        with open(data_file, 'r', encoding="utf-8") as f:
            for line in f:
                a_data = json.loads(line)
                a_datas.append(a_data)
        self.a_datas = a_datas
    def get_results(self,jsondata_list):
        res = {"result": {"results": []}}
        for a_data in jsondata_list:
            type_ = a_data.get("type")
            id_ = a_data.get("id")
            template = self.prompt_templates[type_]
            prompt = ""
            if type_ == "choice":
                choices = a_data["choices"]
                prompt = template.format(Question=a_data["prompt"], A=choices["A"], B=choices["B"], C=choices["C"], D=choices["D"])
            elif type_ == "math":
                prompt = template.format(Question=a_data["prompt"])
            else:
                prompt = template + a_data["prompt"]
            generated_text = []
            outputs = self.llm.generate(prompt, self.sampling_params[type_])
            for output in outputs:
                for o in output.outputs:
                    generated_text.append(o.text)
            generated_text = generated_text[0] if len(generated_text) == 1 else generated_text
            res["result"]["results"].append({"id": id_, "content": generated_text})
        return res

if __name__ == "__main__":
    dataset_path = "A.jsonl"
    comp = Competition()
    comp.load_data(data_file="A.jsonl") #load_data函数加载本地数据集用于验证跑通流程，可使用A榜数据集测试。
    res = comp.get_results(comp.a_datas[:10])
    print(res)
