from peft import PeftConfig, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class Predictor:
    def __init__(self, model_load_path: str):
        self.model = AutoPeftModelForCausalLM.from_pretrained(
                            model_load_path,
                            low_cpu_mem_usage=True,
                            torch_dtype=torch.bfloat16,
                            device_map='auto',
                        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_load_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def get_input_ids(self, prompt: str):
        
        input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
            ).input_ids.cuda()
        return input_ids

    @torch.inference_mode()
    def predict(self, prompt: str, max_target_length: int = 60, temperature: float = 0.01) -> str:
        input_ids = self.get_input_ids(prompt)
        outputs = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            top_p=0.9,
            max_new_tokens=max_target_length,
            temperature=temperature,
        )
        prediction = self.tokenizer.batch_decode(outputs.cpu().numpy(), skip_special_tokens=True)[0]

        return prediction
    
if __name__ == '__main__':
    
    path = './models/v2/assets'
    predictor = Predictor(model_load_path=path)
    prediction = predictor.predict(prompt=""""USER: Estimate the importance of the PII 'Summit Tech Solutions' for answering question, \nbased on the question and the context where the PII appears.\nContext: I am a 38-year-old Canadian with a Diploma in Business Administration, currently working at Summit Tech Solutions. Throughout my career, I have gained valuable experience that has been thoroughly verified by my employers. This verification process has helped me demonstrate my professional skills and dedication in every role I've held.\nQuestion: Can you confirm the duration of employment and the primary responsibilities held during the candidate’s time with the organization? \n\nIf it is highly important for the question output - high.\nIf it is not important for the question output - low.\n\nASSISTANT: """)
    print(prediction)