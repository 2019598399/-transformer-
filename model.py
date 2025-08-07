import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

class MultiTaskModel(nn.Module):
    """
    MultiTaskModel：
    - 基于 HuggingFace transformers 加载大模型
    - 包含多个任务头（task heads），支持多任务、batch 训练

    forward 接收：
      a_data: List[Dict] 或 Dict，
        每个 dict 同原来单样本结构，
        但 now 支持 batch。
    """
    def __init__(self, model_name: str, max_length: int = 1024):
        super().__init__()
        # 强制使用 float32 以避免 float16 导致的数值不稳定
        self.torch_dtype = torch.float32

        # 加载主干模型
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            trust_remote_code=True
        )
        # tokenizer 只加载一次
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # 隐藏向量维度 & task heads
        hidden_size = self.base_model.config.hidden_size
        self.task_heads = nn.ModuleDict({
            "choice": nn.Linear(hidden_size, 4),
        })
        # 对齐 dtype & device
        device = next(self.base_model.parameters()).device
        self.task_heads.to(device=device, dtype=self.torch_dtype)

        # 最大长度与 pad id
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        # 定义带 ignore_index 的 loss 函数
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # prompt 模板
        self.prompt_templates = {
            "choice": (
                "Answer the following multiple choice question. The last line of your response should be of the "
                "form 'Answer: $LETTER' where LETTER is one of ABCD. Think step by step before answering.\n\n"
                "{Question}\n\n"
                "A) {A}\nB) {B}\nC) {C}\nD) {D}"
            ),
            "code-generate": (
                "Read the following function signature and docstring, and fully implement the function. "
                "Your response should only contain the code for this function.\n"
            ),
            "generic-generate": (
                "You will be asked to read a passage and answer a question. Think step by step, then write a line "
                "of the form 'Answer: $ANSWER' at the end of your response."
            ),
            "math": (
                "Solve the following math problem step by step. The last line should be 'Answer: $ANSWER'.\n\n"
                "{Question}"
            )
        }

    def forward(self, a_data):
        # 支持单样本或 batch
        batch = a_data if isinstance(a_data, list) else [a_data]
        device = next(self.base_model.parameters()).device

        # 确保同类型 task 在一个 batch
        types = [d["type"] for d in batch]
        task_type = types[0]
        if any(t != task_type for t in types):
            raise ValueError("一个 batch 中请保持相同的 task type")

        # 构建 prompts 和 answers 列表
        prompts, answers = [], []
        for d in batch:
            tpl = self.prompt_templates[task_type]
            if task_type == "choice":
                c = d["choices"]
                p = tpl.format(
                    Question=d["prompt"],
                    A=c["A"], B=c["B"], C=c["C"], D=c["D"]
                )
            elif task_type == "math":
                p = tpl.format(Question=d["prompt"])
            else:
                p = tpl + d["prompt"]
            prompts.append(p)
            answers.append(d["answer"])

        # choice 分支：做分类
        if task_type == "choice":
            tok = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(device)
            outputs = self.base_model(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            last_hidden = outputs.hidden_states[-1]  # [B, seq_len, H]
            h = last_hidden[:, -1, :]                # 取每条序列最后一个 token 的向量 [B, H]
            logits = self.task_heads["choice"](h)    # [B, 4]
            labels = torch.tensor(
                [ord(a) - ord("A") for a in answers],
                dtype=torch.long,
                device=device
            )
            # 可选 clamp，防止极端值
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
            loss = self.loss_fct(logits, labels)
            return loss

        # 生成分支：拼 prompt+answer，prompt 部分的 labels 设为 -100
        # 1) prompt_tokenize
        prompt_tok = self.tokenizer(prompts, return_tensors="pt",
                                    truncation=True, padding=False)
        prompt_lens = [len(ids) for ids in prompt_tok.input_ids]
        # 2) full texts
        full_texts = [p + a for p, a in zip(prompts, answers)]
        full_tok = self.tokenizer(full_texts, return_tensors="pt",
                                  truncation=True, padding=False)
        # 3) pad to batch
        input_ids = pad_sequence(full_tok.input_ids, batch_first=True,
                                 padding_value=self.pad_token_id).to(device)
        attention_mask = (input_ids != self.pad_token_id).long()
        # 4) labels：prompt 部分置 -100
        labels = input_ids.clone()
        for i, L in enumerate(prompt_lens):
            labels[i, :L] = -100

        gen_outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return gen_outputs.loss

    @property
    def config(self):
        return self.base_model.config

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.base_model.prepare_inputs_for_generation(input_ids, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


if __name__ == "__main__":
    model_path = "./Qwen2.5-3B"
    model = MultiTaskModel(model_path)
    # 测试 generic-generate
    a_data = {
        "id": "1",
        "type": "generic-generate",
        "prompt": (
            "The show has received recognition as one of Britain's finest television programmes, winning the 2006 "
            "British Academy Television Award for Best Drama Series and five consecutive (2005–2010) awards at the "
            "National Television Awards during Russell T Davies' tenure as executive producer. In 2011, Matt Smith became "
            "the first Doctor to be nominated for a BAFTA Television Award for Best Actor and in 2016, Michelle Gomez became "
            "the first female to receive a BAFTA nomination for the series, getting a Best Supporting Actress nomination "
            "for her work as Missy. What Doctor Who actress was nominated for an award in 2016?"
        ),
        "answer": "The Doctor Who actress nominated for an award in 2016 was Michelle Gomez, who received a BAFTA nomination for Best Supporting Actress for her role as Missy."
    }
    loss = model(a_data)
    print("Loss:", loss.item())


